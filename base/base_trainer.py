from abc import abstractmethod

import torch

from logger import TensorboardWriter
from utils import get_module_attr


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, data_loss, data_loss_scale_prior, data_loss_proportion_prior,
                 reg_loss, reg_loss_loc_prior, reg_loss_scale_prior,
                 entropy_loss, transformation_model, registration_module, config):
        self.config = config
        self.checkpoint_dir = config.save_dir
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.transformation_model = transformation_model.to(self.device)
        self.registration_module = registration_module.to(self.device)

        self.data_loss = data_loss.to(self.device)
        self.data_loss_scale_prior = data_loss_scale_prior.to(self.device)
        self.data_loss_proportion_prior = data_loss_proportion_prior.to(self.device)

        self.reg_loss = reg_loss.to(self.device)
        self.reg_loss_loc_prior = reg_loss_loc_prior.to(self.device)
        self.reg_loss_scale_prior = reg_loss_scale_prior.to(self.device)

        self.entropy_loss = entropy_loss.to(self.device)

        if len(device_ids) > 1:
            self.transformation_model = torch.nn.DataParallel(transformation_model, device_ids=device_ids)
            self.registration_module = torch.nn.DataParallel(registration_module, device_ids=device_ids)

            self.data_loss = torch.nn.DataParallel(data_loss, device_ids=device_ids)
            self.data_loss_scale_prior = torch.nn.DataParallel(data_loss_scale_prior, device_ids=device_ids)
            self.data_loss_proportion_prior = torch.nn.DataParallel(data_loss_proportion_prior, device_ids=device_ids)

            self.reg_loss = torch.nn.DataParallel(reg_loss, device_ids=device_ids)
            self.reg_loss_loc_prior = torch.nn.DataParallel(reg_loss_loc_prior, device_ids=device_ids)
            self.reg_loss_scale_prior = torch.nn.DataParallel(reg_loss_scale_prior, device_ids=device_ids)

            self.entropy_loss = torch.nn.DataParallel(entropy_loss, device_ids=device_ids)

        self.reg_loss_type = self.reg_loss.__class__.__name__
        self.diff_op = get_module_attr(self.reg_loss, 'diff_op')  # for use with the transformation Jacobian

        # training logic
        cfg_trainer = config['trainer']

        self.no_iters_VI = int(cfg_trainer['no_iters_VI'])
        self.no_samples_VI_test = int(cfg_trainer['no_samples_VI_test'])

        self.no_iters_burn_in = int(cfg_trainer['no_iters_burn_in'])
        self.no_samples_MCMC = int(cfg_trainer['no_samples_MCMC'])

        self.log_period_VI = cfg_trainer['log_period_VI']
        self.log_period_MCMC = cfg_trainer['log_period_MCMC']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer.write_hparams(config)

    @abstractmethod
    def _train_epoch(self):
        """
        training logic for an epoch
        """

        raise NotImplementedError

    def train(self):
        """
        full training logic
        """

        self._train_epoch()

    def _prepare_device(self, n_gpu_use):
        """
        set up GPU device if available and move model into configured device
        """

        n_gpu = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))

        return device, list_ids
