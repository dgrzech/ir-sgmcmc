from datetime import datetime

import numpy as np
import torch

from base import BaseTrainer
from logger import log_displacement_mean_and_std_dev, log_fields, log_hist_res, log_images, log_sample, \
    save_displacement_mean_and_std_dev, save_fixed_im, save_fixed_mask, save_moving_im, save_moving_mask, save_sample, \
    save_variational_posterior_mean
from utils import SGLD, SobolevGrad, Sobolev_kernel_1D, \
    add_noise_uniform_field, calc_posterior_statistics,\
    calc_VD_factor, calc_metrics, calc_no_non_diffeomorphic_voxels, calc_norm, max_field_update,\
    rescale_residuals, sample_q_v


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

        self.optimizer_q_v = self.config.init_optimizer_q_v(var_params_q_v)

    def __init_optimizer_GMM(self):
        self.optimizer_GMM = self.config.init_optimizer_GMM(self.losses['data']['loss'])

    def __init_optimizer_reg(self):
        reg_loss = self.losses['reg']['loss']

        if reg_loss.learnable:
            self.optimizer_reg = self.config.init_optimizer_reg(reg_loss)

    def __init_optimizer_SG_MCMC(self):
        self.optimizer_SG_MCMC = self.config.init_obj('optimizer_SG_MCMC', torch.optim, [self.v_curr_state])

    def _init_optimizers(self):
        self.optimizer_q_v, self.optimizer_SG_MCMC = None, None

        self.__init_optimizer_GMM()
        self.__init_optimizer_reg()

    def _step_GMM(self, residuals, alpha=1.0):
        data_loss = self.losses['data']['loss']

        data_term = data_loss(residuals.detach()).sum() * alpha
        data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
        data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

        self.optimizer_GMM.zero_grad()
        data_term.backward()  # backprop
        self.optimizer_GMM.step()

    def __calc_sample_loss_VI(self, data_loss, reg_loss, entropy_loss, fixed, moving,
                              var_params_q_v, v_sample_unsmoothed):
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding)
        transformation, displacement = self.transformation_module(v_sample)

        with torch.no_grad():
            no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)

        if self.add_noise_uniform:
            transformation = add_noise_uniform_field(transformation, self.alpha)

        im_moving_warped = self.registration_module(moving['im'], transformation)

        output = {'displacement': displacement, 'transformation': transformation,
                  'im_moving_warped': im_moving_warped, 'log_det_J': log_det_J}

        residuals = data_loss.map(fixed['im'], im_moving_warped)
        alpha = self.__get_VD_factor(residuals, fixed['mask'], data_loss)
        residuals_masked = residuals[fixed['mask']]

        self._step_GMM(residuals_masked, alpha)

        data_term = data_loss(residuals_masked).sum() * alpha
        reg_term, log_y = reg_loss(v_sample)
        reg_term = reg_term.sum()
        entropy_term = entropy_loss(sample=v_sample_unsmoothed, mu=var_params_q_v['mu'], log_var=var_params_q_v['log_var'], log_u=var_params_q_v['log_u']).sum()

        aux = {'alpha': alpha, 'reg_energy': log_y.exp(), 'no_non_diffeomorphic_voxels': no_non_diffeomorphic_voxels, 'residuals': residuals_masked}
        loss_terms = {'data': data_term, 'reg': reg_term, 'entropy': entropy_term}

        if reg_loss.learnable:
            if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
                reg_term_loc_prior = self.losses['reg']['loc_prior'](log_y).sum()
                loss_terms['reg_loc_prior'] = reg_term_loc_prior
            elif reg_loss.__class__.__name__ == 'RegLoss_L2':
                reg_term_w_reg_prior = self.losses['reg']['w_reg_prior'](reg_loss.log_w_reg)
                loss_terms['w_reg_prior'] = reg_term_w_reg_prior

        return loss_terms, output, aux

    def _run_VI(self, fixed, moving, var_params_q_v):
        self.__init_optimizer_q_v(var_params_q_v)
        data_loss, reg_loss, entropy_loss = self.losses['data']['loss'], self.losses['reg']['loss'], self.losses['entropy']

        # .nii.gz/.vtk
        with torch.no_grad():
            save_fixed_im(self.save_dirs, self.im_spacing, fixed['im'])
            save_fixed_mask(self.save_dirs, self.im_spacing, fixed['mask'])
            save_moving_im(self.save_dirs, self.im_spacing, moving['im'])
            save_moving_mask(self.save_dirs, self.im_spacing, moving['mask'])

        for iter_no in range(self.start_iter_VI, self.no_iters_VI + 1):
            # needed to calculate the maximum update in terms of the L2 norm
            var_params_q_v_prev = {k: v.detach().clone() for k, v in var_params_q_v.items()}
            v_sample1_unsmoothed, v_sample2_unsmoothed = sample_q_v(var_params_q_v, no_samples=2)

            loss_terms1, output, aux = self.__calc_sample_loss_VI(data_loss, reg_loss, entropy_loss, fixed, moving, var_params_q_v, v_sample1_unsmoothed)
            loss_terms2, _, _ = self.__calc_sample_loss_VI(data_loss, reg_loss, entropy_loss, fixed, moving, var_params_q_v, v_sample2_unsmoothed)

            # data
            data_term = (loss_terms1['data'] + loss_terms2['data']) / 2.0
            data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
            data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

            # regularisation
            reg_term = (loss_terms1['reg'] + loss_terms2['reg']) / 2.0

            if reg_loss.learnable:
                if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
                    reg_term -= (loss_terms1['reg_loc_prior'] + loss_terms2['reg_loc_prior']) / 2.0
                    reg_term -= self.losses['reg']['scale_prior'](reg_loss.log_scale).sum()
                elif reg_loss.__class__.__name__ == 'RegLoss_L2':
                    reg_term -= (loss_terms1['w_reg_prior'] + loss_terms2['w_reg_prior']) / 2.0

            # entropy
            entropy_term = (loss_terms1['entropy'] + loss_terms2['entropy']) / 2.0
            entropy_term += entropy_loss(log_var=var_params_q_v['log_var'], log_u=var_params_q_v['log_u']).sum()

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
                    self.metrics.update(f'VI/train/GMM/scale_{idx}', data_loss.scales[idx].item())
                    self.metrics.update(f'VI/train/GMM/proportion_{idx}', data_loss.proportions[idx].item())

                if reg_loss.learnable:
                    if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
                        self.metrics.update('VI/train/reg/loc', reg_loss.loc.item())
                        self.metrics.update('VI/train/reg/scale', reg_loss.scale.item())
                    elif reg_loss.__class__.__name__ == 'RegLoss_L2':
                        self.metrics.update('VI/train/reg/w_reg', reg_loss.log_w_reg.exp().item())

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

                for key in var_params_q_v:
                    max_update, max_update_idx = max_field_update(var_params_q_v_prev[key], var_params_q_v[key])
                    self.metrics.update(f'VI/train/max_updates/{key}', max_update.item())

                if iter_no % self.log_period_VI == 0 or iter_no == self.no_iters_VI:
                    # metrics
                    seg_moving_warped = self.registration_module(moving['seg'], output['transformation'])
                    ASD, DSC = calc_metrics(fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing)

                    for structure_idx, structure in enumerate(self.structures_dict):
                        ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
                        self.metrics.update(f'VI/train/ASD/{structure}', ASD_val)
                        self.metrics.update(f'VI/train/DSC/{structure}', DSC_val)

                    # visualisation in tensorboard
                    var_params_q_v_smoothed = self.__get_var_params_smoothed(var_params_q_v)

                    log_hist_res(self.writer, aux['residuals'], data_loss, model='VI')
                    log_images(self.writer, fixed['im'], moving['im'], output['im_moving_warped'])
                    log_fields(self.writer, var_params_q_v_smoothed, output['displacement'], output['log_det_J'])

    @torch.no_grad()
    def _test_VI(self, fixed, moving, var_params_q_v):
        """
        metrics
        """

        samples = torch.zeros([self.no_samples_VI_test, 3, *self.dims])  # samples used in the evaluation

        for test_sample_no in range(1, self.no_samples_VI_test + 1):
            self.writer.set_step(test_sample_no)

            v_sample = sample_q_v(var_params_q_v, no_samples=1)
            v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding)
            transformation, displacement = self.transformation_module(v_sample_smoothed)
            samples[test_sample_no - 1] = displacement.clone().cpu()

            no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)
            self.metrics.update('VI/test/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels)

            im_moving_warped = self.registration_module(moving['im'], transformation)
            seg_moving_warped = self.registration_module(moving['seg'], transformation)

            ASD, DSC = calc_metrics(fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing)

            for structure_idx, structure in enumerate(self.structures_dict):
                ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
                self.metrics.update(f'VI/test/ASD/{structure}', ASD_val)
                self.metrics.update(f'VI/test/DSC/{structure}', DSC_val)

            save_sample(self.save_dirs, self.im_spacing, test_sample_no, im_moving_warped, displacement, log_det_J, 'VI')

        self.logger.info('\nsaving the displacement and warped moving image corresponding to the mean of the approximate variational posterior..')

        mu_v = var_params_q_v['mu']
        mu_v_smoothed = SobolevGrad.apply(mu_v, self.S, self.padding)
        transformation, displacement = self.transformation_module(mu_v_smoothed)
        im_moving_warped = self.registration_module(moving['im'], transformation)

        save_variational_posterior_mean(self.save_dirs, self.im_spacing, im_moving_warped, displacement)

        self.logger.info('\ncalculating sample standard deviation of the displacement..')

        mean, std_dev = calc_posterior_statistics(samples)
        log_displacement_mean_and_std_dev(self.writer, mean, std_dev, 'VI')
        save_displacement_mean_and_std_dev(self.logger, self.save_dirs, self.im_spacing,
                                           mean, std_dev, moving['mask'], 'VI')

        """
        speed
        """

        no_samples_VI_speed_test = 100
        start = datetime.now()

        for VI_test_sample_no in range(1, no_samples_VI_speed_test + 1):
            v_sample = sample_q_v(var_params_q_v, no_samples=1)
            v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding)

            transformation, displacement = self.transformation_module(v_sample_smoothed)
            im_moving_warped = self.registration_module(moving['im'], transformation)
            seg_moving_warped = self.registration_module(moving['seg'], transformation)

        stop = datetime.now()
        VI_sampling_speed = no_samples_VI_speed_test / (stop - start).total_seconds()
        self.logger.info(f'\nVI sampling speed: {VI_sampling_speed:.2f} samples/sec')

    def _SGLD_transition(self, fixed, moving, data_loss, reg_loss):
        curr_state = SGLD.apply(self.v_curr_state, self.SGLD_params['sigma'], self.SGLD_params['tau'])
        curr_state_smoothed = SobolevGrad.apply(curr_state, self.S, self.padding)
        transformation, displacement = self.transformation_module(curr_state_smoothed)

        if self.add_noise_uniform:
            transformation_with_uniform_noise = add_noise_uniform_field(transformation, self.alpha)
            im_moving_warped = self.registration_module(moving['im'], transformation_with_uniform_noise)
        else:
            im_moving_warped = self.registration_module(moving['im'], transformation)

        output = {'im_moving_warped': im_moving_warped.detach().clone(),
                  'displacement': displacement.detach().clone(),
                  'transformation': transformation.detach().clone(),
                  'curr_state': curr_state_smoothed.detach().clone()}

        residuals = data_loss.map(fixed['im'], im_moving_warped)
        residuals_masked = residuals[fixed['mask']].view(self.no_chains, -1)

        data_term = 0.0
        reg_term, log_y = reg_loss(curr_state_smoothed)

        loss_terms = {'data': list(), 'reg': list()}
        aux = {'residuals': residuals_masked, 'alpha': list(), 'reg_energy': list()}

        for idx in range(self.no_chains):
            alpha = self.__get_VD_factor(residuals[idx].unsqueeze(0), fixed['mask'][idx].unsqueeze(0), data_loss)
            self._step_GMM(residuals_masked[idx].unsqueeze(0), alpha)

            chain_data_term = data_loss(residuals_masked[idx]).sum() * alpha
            data_term += chain_data_term

            loss_terms['data'].append(chain_data_term)
            loss_terms['reg'].append(reg_term[idx])

            aux['alpha'].append(alpha)
            aux['reg_energy'].append(log_y[idx].exp())

        data_term -= self.losses['data']['scale_prior'](data_loss.log_scales).sum()
        data_term -= self.losses['data']['proportion_prior'](data_loss.log_proportions).sum()

        reg_term = reg_term.sum()

        if reg_loss.learnable:
            if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
                reg_term -= self.losses['reg']['loc_prior'](log_y).sum()
                reg_term -= self.losses['reg']['scale_prior'](reg_loss.log_scale).sum()
            elif reg_loss.__class__.__name__ == 'RegLoss_L2':
                reg_term -= self.losses['reg']['w_reg_prior'](reg_loss.log_w_reg)

        # total loss
        loss = data_term + reg_term

        self.optimizer_SG_MCMC.zero_grad()

        if reg_loss.learnable:
            self.optimizer_reg.zero_grad()

        loss.backward()  # backprop

        self.optimizer_SG_MCMC.step()

        if reg_loss.learnable:
            self.optimizer_reg.step()

        return loss_terms, output, aux

    def _run_MCMC(self, fixed, moving, var_params_q_v):
        data_loss, reg_loss = self.losses['data']['loss'], self.losses['reg']['loss']

        fixed = {k: v.expand(self.no_chains, *v.shape[1:]) for k, v in fixed.items()}
        moving = {k: v.expand(self.no_chains, *v.shape[1:]) for k, v in moving.items()}
        self.__SGLD_init(var_params_q_v)

        no_samples = self.no_chains * self.no_samples_MCMC // self.log_period_MCMC
        samples = torch.zeros([no_samples, 3, *self.dims])  # samples used in the evaluation
        test_sample_idx = 0

        self.logger.info(f'\nNO. CHAINS: {self.no_chains}, BURNING IN...')

        for sample_no in range(1, self.no_iters_burn_in + self.no_samples_MCMC + 1):
            if sample_no < self.no_iters_burn_in and sample_no % self.log_period_MCMC == 0:
                self.logger.info(f'burn-in sample no. {sample_no}/{self.no_iters_burn_in}')

            """
            stochastic gradient Langevin dynamics
            """

            loss_terms, output, aux = self._SGLD_transition(fixed, moving, data_loss, reg_loss)

            if sample_no == self.no_iters_burn_in:
                self.logger.info('ENDED BURNING IN')

            """
            outputs
            """

            with torch.no_grad():
                self.writer.set_step(sample_no)

                if self.no_samples_MCMC < 1e4 or (sample_no - 1) % 100 == 0:  # NOTE (DG): the logs are too large otherwise
                    # model parameters
                    for idx in range(data_loss.no_components):
                        self.metrics.update(f'MCMC/GMM/scale_{idx}', data_loss.scales[idx].item())
                        self.metrics.update(f'MCMC/GMM/proportion_{idx}', data_loss.proportions[idx].item())

                    if reg_loss.learnable:
                        if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
                            self.metrics.update('MCMC/reg/loc', reg_loss.loc.item())
                            self.metrics.update('MCMC/reg/scale', reg_loss.scale.item())
                        elif reg_loss.__class__.__name__ == 'RegLoss_L2':
                            self.metrics.update('MCMC/reg/w_reg', reg_loss.log_w_reg.exp().item())

                    # losses
                    total_loss = sum(loss_terms['data']) + sum(loss_terms['reg'])
                    self.metrics.update(f'MCMC/avg_loss', total_loss.item() / self.no_chains)

                    for idx in range(self.no_chains):
                        self.metrics.update(f'MCMC/chain_{idx}/data_term', loss_terms['data'][idx].item())
                        self.metrics.update(f'MCMC/chain_{idx}/reg_term', loss_terms['reg'][idx].item())
                        self.metrics.update(f'MCMC/chain_{idx}/VD/alpha', aux['alpha'][idx].item())
                        self.metrics.update(f'MCMC/chain_{idx}/reg/energy', aux['reg_energy'][idx].item())

                if sample_no > self.no_iters_burn_in:
                    if sample_no % self.log_period_MCMC == 0 or sample_no == self.no_samples_MCMC:
                        self.writer.set_step(sample_no - self.no_iters_burn_in)

                        displacement, transformation = output['displacement'], output['transformation']
                        im_moving_warped = output['im_moving_warped']
                        seg_moving_warped = self.registration_module(moving['seg'], transformation)

                        ASD, DSC = calc_metrics(fixed['seg'], seg_moving_warped, self.structures_dict, self.im_spacing, no_samples=self.no_chains)
                        no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)

                        v_curr_state_smoothed = output['curr_state']
                        v_norm, displacement_norm = calc_norm(v_curr_state_smoothed), calc_norm(displacement)

                        for idx in range(self.no_chains):
                            samples[test_sample_idx] = displacement[idx].detach().cpu()
                            test_sample_idx += 1

                            for structure_idx, structure in enumerate(self.structures_dict):
                                ASD_val, DSC_val = ASD[idx][structure_idx], DSC[idx][structure_idx]
                                self.metrics.update(f'MCMC/chain_{idx}/ASD/{structure}', ASD_val)
                                self.metrics.update(f'MCMC/chain_{idx}/DSC/{structure}', DSC_val)

                            no_non_diffeomorphic_voxels_chain = no_non_diffeomorphic_voxels[idx]
                            self.metrics.update(f'MCMC/chain_{idx}/no_non_diffeomorphic_voxels',
                                                no_non_diffeomorphic_voxels_chain)

                            if no_non_diffeomorphic_voxels_chain > 0.001 * self.no_voxels:
                                self.logger.info(f'chain {idx}, sample {sample_no}: '
                                                 f'detected {no_non_diffeomorphic_voxels} voxels where '
                                                 f'the sampled transformation is not diffeomorphic; exiting..')
                                exit()

                            residuals = aux['residuals'][idx]

                            # tensorboard
                            log_sample(self.writer, idx, im_moving_warped, v_norm, displacement_norm, log_det_J)
                            log_hist_res(self.writer, residuals, data_loss, model='MCMC', chain_no=idx)

                            # .nii.gz/.vtk
                            save_sample(self.save_dirs, self.im_spacing, sample_no, im_moving_warped, displacement, log_det_J, 'MCMC', chain_no=idx)

        self.logger.info('\ncalculating sample standard deviation of the displacement..')

        mean, std_dev = calc_posterior_statistics(samples)
        log_displacement_mean_and_std_dev(self.writer, mean, std_dev, 'MCMC')
        save_displacement_mean_and_std_dev(self.logger, self.save_dirs, self.im_spacing,
                                           mean, std_dev, moving['mask'], 'MCMC')

        """
        speed
        """

        no_samples_MCMC_speed_test = 100
        start = datetime.now()

        for sample_no in range(1, no_samples_MCMC_speed_test + 1):
            _, output, _ = self._SGLD_transition(fixed, moving, data_loss, reg_loss)
            seg_moving_warped = self.registration_module(moving['seg'], output['transformation'])

        stop = datetime.now()
        MCMC_sampling_speed = self.no_chains * no_samples_MCMC_speed_test / (stop - start).total_seconds()
        self.logger.info(f'\nMCMC sampling speed: {MCMC_sampling_speed:.2f} samples/sec')

    def _run_model(self):
        for batch_idx, (fixed, moving, var_params_q_v) in enumerate(self.data_loader):
            self.__data_init(fixed, moving, var_params_q_v)
            self.__GMM_init(fixed, moving, var_params_q_v)
            self.__metrics_init(fixed, moving)

            """
            VI
            """

            if self.VI:
                # fit the approximate variational posterior
                start = datetime.now()
                self._run_VI(fixed, moving, var_params_q_v)
                stop = datetime.now()

                VI_time = (stop - start).total_seconds()
                self.logger.info(f'VI took {VI_time:.2f} seconds')

                # sample from it
                self._test_VI(fixed, moving, var_params_q_v)

            """
            MCMC
            """

            if self.MCMC:
                self._run_MCMC(fixed, moving, var_params_q_v)

    def __get_VD_factor(self, residuals, mask, data_loss):
        if self.virutal_decimation:
            residuals_rescaled = rescale_residuals(residuals.detach(), mask, data_loss)
            alpha = calc_VD_factor(residuals_rescaled, mask)
        else:
            alpha = 1.0

        return alpha

    @torch.no_grad()
    def __data_init(self, fixed, moving, var_params_q_v):
        for key in fixed:
            fixed[key] = fixed[key].to(self.device)
            moving[key] = moving[key].to(self.device)

        for param_key in var_params_q_v:
            var_params_q_v[param_key] = var_params_q_v[param_key].to(self.device)
            var_params_q_v[param_key].requires_grad_(True)

        self.dims, self.im_spacing = self.data_loader.dims, self.data_loader.im_spacing
        self.no_voxels = np.prod(self.dims)

    def __GMM_init(self, fixed, moving, var_params_q_v):
        data_loss = self.losses['data']['loss']

        v_sample_unsmoothed = sample_q_v(var_params_q_v)
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding)
        transformation, displacement = self.transformation_module(v_sample)

        im_moving_warped = self.registration_module(moving['im'], transformation)
        residuals = data_loss.map(fixed['im'], im_moving_warped)
        residuals_masked = residuals[fixed['mask']]
        residuals_std_dev = torch.std(residuals_masked)

        data_loss.init_parameters(residuals_std_dev)
        alpha = self.__get_VD_factor(residuals, fixed['mask'], data_loss)

        no_steps_GMM_warm_up = 25

        for _ in range(1, no_steps_GMM_warm_up + 1):
            self._step_GMM(residuals_masked, alpha)

    @torch.no_grad()
    def __metrics_init(self, fixed, moving):
        step = 0
        self.writer.set_step(step)

        residuals = self.losses['data']['loss'].map(fixed['im'], moving['im'])
        residuals_masked = residuals[fixed['mask']]

        log_hist_res(self.writer, residuals_masked, self.losses['data']['loss'], model='VI')
        log_images(self.writer, fixed['im'], moving['im'], moving['im'])

        # metrics
        ASD, DSC = calc_metrics(fixed['seg'], moving['seg'], self.structures_dict, self.im_spacing)

        for structure_idx, structure in enumerate(self.structures_dict):
            ASD_val, DSC_val = ASD[0][structure_idx], DSC[0][structure_idx]
            self.metrics.update(f'VI/train/ASD/{structure}', ASD_val)
            self.metrics.update(f'VI/train/DSC/{structure}', DSC_val)

    @torch.no_grad()
    def __Sobolev_gradients_init(self):
        _s = self.config['Sobolev_grad']['s']
        _lambda = self.config['Sobolev_grad']['lambda']
        padding_sz = _s

        S, S_sqrt = Sobolev_kernel_1D(_s, _lambda)
        S = torch.from_numpy(S).float().unsqueeze(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2).to(self.device)
        S_y = S.unsqueeze(2).unsqueeze(4).to(self.device)
        S_z = S.unsqueeze(3).unsqueeze(4).to(self.device)

        self.padding = (padding_sz, ) * 6
        self.S = {'x': S_x, 'y': S_y, 'z': S_z}

    @torch.no_grad()
    def __SGLD_init(self, var_params_q_v):
        if self.MCMC_init == 'VI':
            v_curr_state = torch.empty([self.no_chains, 3, *var_params_q_v['mu'].shape[2:]], device=self.device)

            for idx in range(self.no_chains):
                v_curr_state[idx] = sample_q_v(var_params_q_v, no_samples=1).detach()

            log_var_v = var_params_q_v['log_var'].detach().clone()
            sigma = torch.exp(0.5 * log_var_v).expand_as(v_curr_state)

        elif self.MCMC_init in ['identity', 'noise']:
            if self.MCMC_init == 'identity':
                v_curr_state = torch.zeros([self.no_chains, 3, *var_params_q_v['mu'].shape[2:]], device=self.device)
            elif self.MCMC_init == 'noise':
                v_curr_state = torch.randn([self.no_chains, 3, *var_params_q_v['mu'].shape[2:]], device=self.device)

            sigma = torch.ones_like(v_curr_state)

        v_curr_state.requires_grad_(True)
        sigma.requires_grad_(False)

        tau = self.config['optimizer_SG_MCMC']['args']['lr']

        self.SGLD_params = {'sigma': sigma, 'tau': tau}
        self.v_curr_state = v_curr_state
        self.__init_optimizer_SG_MCMC()

    def __get_var_params_smoothed(self, var_params):
        return {k: SobolevGrad.apply(v, self.S, self.padding) for k, v in var_params.items()}
