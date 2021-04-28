from datetime import datetime

import numpy as np
import torch

from base import BaseTrainer
from logger import log_fields, log_hist_res, log_images, log_sample, save_displacement_mean_and_std_dev, \
    save_fixed_im, save_fixed_mask, save_moving_im, save_sample
from optimizers import Adam
from utils import SGLD, SobolevGrad, Sobolev_kernel_1D, add_noise_uniform_field, calc_VD_factor, \
    calc_displacement_std_dev, calc_metrics, \
    calc_no_non_diffeomorphic_voxels, max_field_update, rescale_residuals, sample_q_v


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, config, data_loader, losses, transformation_module, registration_module, metrics):
        super().__init__(config, data_loader, losses, transformation_module, registration_module, metrics)

        # optimizers
        self._init_optimizers()

        # Sobolev gradients
        self.Sobolev_grad = config['Sobolev_grad']['enabled']

        if self.Sobolev_grad:
            self.__Sobolev_gradients_init()

        # uniform noise magnitude
        cfg_trainer = config['trainer']

        self.add_noise_uniform = cfg_trainer['uniform_noise']['enabled']

        if self.add_noise_uniform:
            self.alpha = cfg_trainer['uniform_noise']['magnitude']
        
        # virtual decimation
        self.virutal_decimation = config['virtual_decimation']

    def __init_optimizer_q_v(self, var_params_q_v):
        for param_key in var_params_q_v:
            var_params_q_v[param_key].requires_grad_(True)

        trainable_params_q_v = filter(lambda p: p.requires_grad, var_params_q_v.values())
        self.optimizer_q_v = self.config.init_obj('optimizer_q_v', torch.optim, trainable_params_q_v)

    def __init_optimizer_GMM(self):
        data_loss = self.losses['data']['loss']

        self.optimizer_GMM = Adam(
                [{'params': [data_loss.log_std], 'lr': 1e-1}, {'params': [data_loss.logits], 'lr': 1e-2}],
                lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)

    def __init_optimizer_SG_MCMC(self):
        self.optimizer_SG_MCMC = self.config.init_obj('optimizer_SG_MCMC', torch.optim, [self.v_curr_state])

    def __init_optimizer_reg(self):
        reg_loss = self.losses['reg']['loss']

        if reg_loss.learnable:
            self.optimizer_reg = Adam([{'params': [reg_loss.loc, reg_loss.log_scale]}], lr=1e-1, betas=(0.9, 0.95))

    def _init_optimizers(self):
        self.optimizer_q_v, self.optimizer_SG_MCMC = None, None

        self.__init_optimizer_GMM()
        self.__init_optimizer_reg()

    def _step_GMM(self, res, alpha=1.0):
        data_loss = self.losses['data']['loss']

        data_term = data_loss(res.detach()).sum() * alpha
        data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
        data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

        self.optimizer_GMM.zero_grad()
        data_term.backward()  # backprop
        self.optimizer_GMM.step()

    def __calc_sample_loss_VI(self, data_loss, reg_loss, entropy_loss, moving, var_params_q_v, v_sample_unsmoothed):
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding)
        transformation, displacement = self.transformation_module(v_sample)

        with torch.no_grad():
            no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)

        if self.add_noise_uniform:
            transformation = add_noise_uniform_field(transformation, self.alpha)

        im_moving_warped = self.registration_module(moving['im'], transformation)

        res = data_loss.map(self.fixed['im'], im_moving_warped)
        alpha = self.__get_VD_factor(res, data_loss)
        res_masked = res[self.fixed['mask']]

        self._step_GMM(res_masked, alpha)

        data_term = data_loss(res_masked).sum() * alpha
        reg_term, log_y = reg_loss(v_sample)
        reg_term = reg_term.sum()
        entropy_term = entropy_loss(sample=v_sample_unsmoothed, mu=var_params_q_v['mu'], log_var=var_params_q_v['log_var'], u=var_params_q_v['u']).sum()

        loss_terms = {'data': data_term, 'reg': reg_term, 'entropy': entropy_term}
        output = {'im_moving_warped': im_moving_warped,
                  'transformation': transformation, 'log_det_J': log_det_J, 'displacement': displacement}
        aux = {'alpha': alpha, 'reg_energy': log_y.exp(), 'no_non_diffeomorphic_voxels': no_non_diffeomorphic_voxels,
               'res': res_masked}

        if reg_loss.learnable:
            reg_term_loc_prior = self.losses['reg']['loc_prior'](log_y).sum() if reg_loss.learnable else 0.0
            loss_terms['reg_loc_prior'] = reg_term_loc_prior

        return loss_terms, output, aux

    def _run_VI(self, im_pair_idxs, moving, var_params_q_v):
        self.__init_optimizer_q_v(var_params_q_v)
        data_loss, reg_loss, entropy_loss = self.losses['data']['loss'], self.losses['reg']['loss'], self.losses['entropy']

        # .nii.gz/.vtk
        with torch.no_grad():
            save_fixed_im(self.save_dirs, self.im_spacing, self.fixed['im'])
            save_fixed_mask(self.save_dirs, self.im_spacing, self.fixed['mask'])
            save_moving_im(im_pair_idxs, self.save_dirs, self.im_spacing, moving['im'])

        for iter_no in range(self.start_iter_VI, self.no_iters_VI + 1):
            # needed to calculate the maximum update in terms of the L2 norm
            if iter_no % self.log_period_VI == 0 or iter_no == self.no_iters_VI:
                var_params_q_v_prev = {k: v.detach().clone() for k, v in var_params_q_v.items()}

            v_sample1_unsmoothed, v_sample2_unsmoothed = sample_q_v(var_params_q_v, no_samples=2)

            loss_terms1, output, aux = self.__calc_sample_loss_VI(data_loss, reg_loss, entropy_loss, moving, var_params_q_v, v_sample1_unsmoothed)
            loss_terms2, _, _ = self.__calc_sample_loss_VI(data_loss, reg_loss, entropy_loss, moving, var_params_q_v, v_sample2_unsmoothed)

            # data
            data_term = (loss_terms1['data'] + loss_terms2['data']) / 2.0
            data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
            data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

            # regularisation
            reg_term = (loss_terms1['reg'] + loss_terms2['reg']) / 2.0

            if reg_loss.learnable:
                reg_term -= (loss_terms1['reg_loc_prior'] + loss_terms2['reg_loc_prior']) / 2.0
                reg_term -= self.losses['reg']['scale_prior'](reg_loss.log_scale).sum()

            # entropy
            entropy_term = (loss_terms1['entropy'] + loss_terms2['entropy']) / 2.0
            entropy_term += entropy_loss(log_var=var_params_q_v['log_var'], u=var_params_q_v['u']).sum()

            # total loss
            loss = data_term + reg_term - entropy_term

            if reg_loss.learnable:
                self.optimizer_reg.zero_grad()

            self.optimizer_q_v.zero_grad()

            loss.backward()  # backprop

            if reg_loss.learnable:
                self.optimizer_reg.step()

            self.optimizer_q_v.step()

            """
            tensorboard and logging
            """

            with torch.no_grad():
                self.writer.set_step(iter_no)

                # model parameters
                for idx in range(data_loss.no_components):
                    self.metrics.update('VI/train/GMM/scale_' + str(idx), data_loss.scales[idx].item())
                    self.metrics.update('VI/train/GMM/proportion_' + str(idx), data_loss.proportions[idx].item())

                if reg_loss.learnable:
                    self.metrics.update('VI/train/reg/loc', reg_loss.loc.item())
                    self.metrics.update('VI/train/reg/scale', reg_loss.scale.item())

                if self.virutal_decimation:
                    self.metrics.update('VI/train/VD/alpha', aux['alpha'].item())

                # losses
                self.metrics.update('VI/train/data_term', data_term.item())
                self.metrics.update('VI/train/reg_term', reg_term.item())
                self.metrics.update('VI/train/entropy_term', entropy_term.item())
                self.metrics.update('VI/train/total_loss', loss.item())

                # other
                self.metrics.update('VI/train/reg/energy', aux['reg_energy'].item())
                self.metrics.update('VI/train/no_non_diffeomorphic_voxels', aux['no_non_diffeomorphic_voxels'].item())

                if iter_no % self.log_period_VI == 0 or iter_no == self.no_iters_VI:
                    for key in var_params_q_v:
                        max_update, max_update_idx = max_field_update(var_params_q_v_prev[key], var_params_q_v[key])
                        self.metrics.update('VI/train/max_updates/' + key, max_update.item())

                    # metrics
                    seg_moving_warped = self.registration_module(moving['seg'], output['transformation'])
                    ASD, DSC = calc_metrics(im_pair_idxs, self.fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing)

                    for structure_idx, structure in enumerate(self.structures_dict):
                        ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
                        self.metrics.update('VI/train/ASD/' + structure, ASD_val.item())
                        self.metrics.update('VI/train/DSC/' + structure, DSC_val.item())

                    # visualisation in tensorboard
                    var_params_q_v_smoothed = self.__get_var_params_smoothed(var_params_q_v)

                    log_hist_res(self.writer, im_pair_idxs, aux['res'], data_loss, model='VI')
                    log_images(self.writer, im_pair_idxs, self.fixed['im'], moving['im'], output['im_moving_warped'])
                    log_fields(self.writer, im_pair_idxs, var_params_q_v_smoothed, output['displacement'], output['log_det_J'])

    @torch.no_grad()
    def _test_VI(self, im_pair_idxs, moving, var_params_q_v):
        """
        metrics
        """

        for test_sample_no in range(1, self.no_samples_VI_test + 1):
            self.writer.set_step(test_sample_no)

            v_sample = sample_q_v(var_params_q_v, no_samples=1)
            v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding)
            transformation, displacement = self.transformation_module(v_sample_smoothed)

            try:
                displacement_mean += displacement
            except:
                displacement_mean = displacement.detach().clone()

            no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)
            self.metrics.update('VI/test/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels)

            im_moving_warped = self.registration_module(moving['im'], transformation)
            seg_moving_warped = self.registration_module(moving['seg'], transformation)

            ASD, DSC = calc_metrics(im_pair_idxs, self.fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing)

            for structure_idx, structure in enumerate(self.structures_dict):
                ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
                self.metrics.update('VI/test/ASD/' + structure, ASD_val.item())
                self.metrics.update('VI/test/DSC/' + structure, DSC_val.item())

            save_sample(im_pair_idxs, self.save_dirs, self.im_spacing, test_sample_no, im_moving_warped, displacement, log_det_J, model='VI')

        displacement_mean /= self.no_samples_VI_test
        displacement_mean *= self.fixed['mask']

        self.logger.info('\ncalculating sample standard deviation of the displacement..')
        displacement_std_dev = calc_displacement_std_dev(self.logger, self.save_dirs, displacement_mean, 'VI')
        displacement_std_dev *= self.fixed['mask']

        save_displacement_mean_and_std_dev(self.logger, im_pair_idxs[0], self.save_dirs, self.im_spacing, displacement_mean[0], displacement_std_dev[0], 'VI')

        """
        speed
        """

        start = datetime.now()

        for VI_test_sample_no in range(1, self.no_samples_VI_test + 1):
            v_sample = sample_q_v(var_params_q_v, no_samples=1)
            v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding)

            transformation, displacement = self.transformation_module(v_sample_smoothed)
            im_moving_warped = self.registration_module(moving['im'], transformation)
            seg_moving_warped = self.registration_module(moving['seg'], transformation)

        stop = datetime.now()
        VI_sampling_speed = (self.no_samples_VI_test * 10 + 1) / (stop - start).total_seconds()
        self.logger.info(f'VI sampling speed: {VI_sampling_speed:.2f} samples/sec')

    def _SGLD_transition(self, moving, data_loss, reg_loss):
        v_curr_state = SGLD.apply(self.v_curr_state, self.SGLD_params['sigma'], self.SGLD_params['tau'])
        v_curr_state_smoothed = SobolevGrad.apply(v_curr_state, self.S, self.padding)
        transformation, displacement = self.transformation_module(v_curr_state_smoothed)

        # data
        if self.add_noise_uniform:
            transformation_with_uniform_noise = add_noise_uniform_field(transformation, self.alpha)
            im_moving_warped = self.registration_module(moving['im'], transformation_with_uniform_noise)
        else:
            im_moving_warped = self.registration_module(moving['im'], transformation)

        res = data_loss.map(self.fixed['im'], im_moving_warped)
        alpha = self.__get_VD_factor(res, data_loss)
        res_masked = res[self.fixed['mask']]

        self._step_GMM(res_masked, alpha)

        data_term = data_loss(res_masked).sum() * alpha
        data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
        data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

        # regularisation
        reg_term, log_y = reg_loss(v_curr_state_smoothed)
        reg_term = reg_term.sum()

        if reg_loss.learnable:
            reg_term -= self.losses['reg']['loc_prior'](log_y).sum()
            reg_term -= self.losses['reg']['scale_prior'](reg_loss.log_scale).sum()

        # total loss
        loss = data_term + reg_term

        self.optimizer_SG_MCMC.zero_grad()
        self.optimizer_reg.zero_grad()

        loss.backward()  # backprop

        self.optimizer_SG_MCMC.step()
        self.optimizer_reg.step()

        loss_terms = {'data': data_term, 'reg': reg_term, 'total_loss': loss}
        output = {'v_curr_state_smoothed': v_curr_state_smoothed, 'transformation': transformation, 'displacement': displacement,
                  'im_moving_warped': im_moving_warped}
        aux = {'alpha': alpha, 'reg_energy': log_y.exp(), 'res': res_masked}

        return loss_terms, output, aux

    def _run_MCMC(self, im_pair_idxs, moving, var_params_q_v):
        self.__SGLD_init(var_params_q_v)
        data_loss, reg_loss = self.losses['data']['loss'], self.losses['reg']['loss']

        self.logger.info('\nBURNING IN THE MARKOV CHAIN')

        for sample_no in range(1, self.no_iters_burn_in + self.no_samples_MCMC + 1):
            if sample_no < self.no_iters_burn_in and sample_no % self.log_period_MCMC == 0:
                self.logger.info('burn-in sample no. ' + str(sample_no) + '/' + str(self.no_iters_burn_in))

            """
            stochastic gradient Langevin dynamics
            """

            loss_terms, output, aux = self._SGLD_transition(moving, data_loss, reg_loss)

            if sample_no == self.no_iters_burn_in:
                self.logger.info('ENDED BURNING IN')

            """
            outputs
            """

            with torch.no_grad():
                self.writer.set_step(sample_no)

                # model parameters
                for idx in range(data_loss.no_components):
                    self.metrics.update('MCMC/GMM/scale_' + str(idx), data_loss.scales[idx].item())
                    self.metrics.update('MCMC/GMM/proportion_' + str(idx), data_loss.proportions[idx].item())

                if reg_loss.learnable:
                    self.metrics.update('MCMC/reg/loc', reg_loss.loc.item())
                    self.metrics.update('MCMC/reg/scale', reg_loss.scale.item())

                if self.virutal_decimation:
                    self.metrics.update('MCMC/VD/alpha', aux['alpha'].item())

                # losses
                self.metrics.update('MCMC/data_term', loss_terms['data'].item())
                self.metrics.update('MCMC/reg_term', loss_terms['reg'].item())
                self.metrics.update('MCMC/total_loss', loss_terms['total_loss'].item())

                # other
                self.metrics.update('MCMC/reg/energy', aux['reg_energy'].item())

                if sample_no > self.no_iters_burn_in:
                    if sample_no % self.log_period_MCMC == 0 or sample_no == self.no_samples_MCMC:
                        self.writer.set_step(sample_no - self.no_iters_burn_in)

                        displacement = output['displacement']

                        try:
                            displacement_mean += displacement
                        except:
                            displacement_mean = displacement.detach().clone()

                        transformation = output['transformation']
                        no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)
                        self.metrics.update('MCMC/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels.item())

                        # metrics
                        seg_moving_warped = self.registration_module(moving['seg'], transformation)
                        ASD, DSC = calc_metrics(im_pair_idxs, self.fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing)

                        for structure_idx, structure in enumerate(self.structures_dict):
                            ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
                            self.metrics.update('MCMC/ASD/' + structure, ASD_val.item())
                            self.metrics.update('MCMC/DSC/' + structure, DSC_val.item())

                        v_curr_state_smoothed = output['v_curr_state_smoothed']
                        im_moving_warped = output['im_moving_warped']
                        res_masked = aux['res']

                        # tensorboard
                        log_sample(self.writer, im_pair_idxs, data_loss, res_masked, im_moving_warped, v_curr_state_smoothed, displacement, log_det_J)

                        # .nii.gz/.vtk
                        save_sample(im_pair_idxs, self.save_dirs, self.im_spacing, sample_no, im_moving_warped, displacement, log_det_J, model='MCMC')

                        if no_non_diffeomorphic_voxels > 0.001 * np.prod(self.data_loader.dims):
                            self.logger.info('sample ' + str(sample_no) + ', detected ' + str(no_non_diffeomorphic_voxels) + ' voxels where the sample transformation is not diffeomorphic; exiting..')
                            exit()

        with torch.no_grad():
            no_samples = self.no_samples_MCMC / self.log_period_MCMC
            displacement_mean /= no_samples
            displacement_mean *= self.fixed['mask']

            self.logger.info('\ncalculating sample standard deviation of the displacement..')
            displacement_std_dev = calc_displacement_std_dev(self.logger, self.save_dirs, displacement_mean, 'MCMC')
            displacement_std_dev *= self.fixed['mask']

            save_displacement_mean_and_std_dev(self.logger, im_pair_idxs[0], self.save_dirs, self.im_spacing, displacement_mean[0], displacement_std_dev[0], 'MCMC')

        """
        speed
        """

        no_samples_MCMC_speed_test = 100
        start = datetime.now()

        for sample_no in range(1, no_samples_MCMC_speed_test + 1):
            _, output, _ = self._SGLD_transition(moving, data_loss, reg_loss)
            transformation = output['transformation']
            seg_moving_warped = self.registration_module(moving['seg'], transformation)

        stop = datetime.now()
        MCMC_sampling_speed = no_samples_MCMC_speed_test / (stop - start).total_seconds()
        self.logger.info(f'\nMCMC sampling speed: {MCMC_sampling_speed:.2f} samples/sec')

    def _run_model(self):
        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(self.data_loader):
            im_pair_idxs = im_pair_idxs.tolist()

            self.__moving_init(moving, var_params_q_v)
            self.__GMM_init(moving, var_params_q_v)
            self.__metrics_init(im_pair_idxs, moving, var_params_q_v)

            """
            VI
            """

            if self.VI:
                start = datetime.now()

                # fit the approximate variational posterior
                self._run_VI(im_pair_idxs, moving, var_params_q_v)

                stop = datetime.now()

                VI_time = (stop - start).total_seconds()
                self.logger.info(f'VI took {VI_time:.2f} seconds')

                # sample from it
                self._test_VI(im_pair_idxs, moving, var_params_q_v)

            """
            MCMC
            """

            if self.MCMC:
                self._run_MCMC(im_pair_idxs, moving, var_params_q_v)

    def __get_VD_factor(self, res, data_loss):
        if self.virutal_decimation:
            res_rescaled = rescale_residuals(res.detach(), self.fixed['mask'], data_loss)
            alpha = calc_VD_factor(res_rescaled, self.fixed['mask'])
        else:
            alpha = 1.0

        return alpha

    @torch.no_grad()
    def __moving_init(self, moving, var_params_q_v):
        for key in moving:
            moving[key] = moving[key].to(self.device)

        for param_key in var_params_q_v:
            var_params_q_v[param_key] = var_params_q_v[param_key].to(self.device)
            var_params_q_v[param_key].requires_grad_(True)

    def __GMM_init(self, moving, var_params_q_v):
        data_loss = self.losses['data']['loss']

        v_sample_unsmoothed = sample_q_v(var_params_q_v)
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding)

        transformation, displacement = self.transformation_module(v_sample)
        im_moving_warped = self.registration_module(moving['im'], transformation)

        res = data_loss.map(self.fixed['im'], im_moving_warped)
        res_masked = res[self.fixed['mask']]
        res_std = torch.std(res_masked)

        data_loss.init_parameters(res_std)
        alpha = self.__get_VD_factor(res, data_loss)

        for step in range(1, 51):
            self._step_GMM(res_masked, alpha)

    @torch.no_grad()
    def __metrics_init(self, im_pair_idxs, moving, var_params_q_v):
        step = 0
        self.writer.set_step(step)
        self.__moving_init(moving, var_params_q_v)

        res = self.losses['data']['loss'].map(self.fixed['im'], moving['im'])
        res_masked = res[self.fixed['mask']]

        log_hist_res(self.writer, im_pair_idxs, res_masked, self.losses['data']['loss'], model='VI')
        log_images(self.writer, im_pair_idxs, self.fixed['im'], moving['im'], moving['im'])

        # metrics
        ASD, DSC = calc_metrics(im_pair_idxs, self.fixed['seg'], moving['seg'], self.structures_dict, self.im_spacing)

        for structure_idx, structure in enumerate(self.structures_dict):
            ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
            self.metrics.update('VI/train/ASD/' + structure, ASD_val.item())
            self.metrics.update('VI/train/DSC/' + structure, DSC_val.item())

    @torch.no_grad()
    def __Sobolev_gradients_init(self):
        _s = self.config['Sobolev_grad']['s']
        _lambda = self.config['Sobolev_grad']['lambda']
        padding_sz = _s // 2

        S, S_sqrt = Sobolev_kernel_1D(_s, _lambda)
        S = torch.from_numpy(S).float().unsqueeze(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2).to(self.device)
        S_y = S.unsqueeze(2).unsqueeze(4).to(self.device)
        S_z = S.unsqueeze(3).unsqueeze(4).to(self.device)

        self.padding = (padding_sz,) * 6
        self.S = {'x': S_x, 'y': S_y, 'z': S_z}

    @torch.no_grad()
    def __SGLD_init(self, var_params_q_v):
        if self.MCMC_init == 'VI':
            v_curr_state = sample_q_v(var_params_q_v, no_samples=1).detach()

            log_var_v = var_params_q_v['log_var'].detach().clone()
            sigma = torch.exp(0.5 * log_var_v)
        elif self.MCMC_init in ['identity', 'noise']:
            if self.MCMC_init == 'identity':
                v_curr_state = torch.zeros_like(var_params_q_v['mu']).detach()
            elif self.MCMC_init == 'noise':
                v_curr_state = torch.randn_like(var_params_q_v['mu']).detach()

            sigma = torch.ones_like(v_curr_state)

        v_curr_state.requires_grad_(True)
        sigma.requires_grad_(False)

        cfg_sg_mcmc = self.config['optimizer_SG_MCMC']['args']
        tau = cfg_sg_mcmc['lr']

        self.SGLD_params = {'sigma': sigma, 'tau': tau}
        self.v_curr_state = v_curr_state
        self.__init_optimizer_SG_MCMC()

    def __get_var_params_smoothed(self, var_params):
        return {k: SobolevGrad.apply(v, self.S, self.padding) for k, v in var_params.items()}
