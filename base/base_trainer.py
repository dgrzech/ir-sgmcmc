from abc import abstractmethod

from logger import TensorboardWriter
from utils import MetricTracker


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, losses, transformation_module, registration_module, metrics):
        self.config = config
        self.logger = config.logger

        self.device = 'cuda:0'

        self.data_loader = data_loader
        self.structures_dict = self.config.structures_dict
        self.save_dirs = self.data_loader.save_dirs

        # losses
        self.losses = dict()

        self.losses['data'] = {k: loss.to(self.device) for k, loss in losses['data'].items()}
        self.losses['reg'] = {k: loss.to(self.device) for k, loss in losses['reg'].items()}
        self.losses['entropy'] = losses['entropy'].to(self.device)

        # transformation and registration modules
        self.transformation_module = transformation_module.to(self.device)
        self.registration_module = registration_module.to(self.device)
        
        # differential operator for use with the transformation Jacobian
        self.diff_op = self.losses['reg']['loss'].diff_op

        # model logic
        cfg_trainer = config['trainer']

        self.VI = cfg_trainer['VI']
        self.start_iter_VI, self.no_iters_VI = 1, int(cfg_trainer['no_iters_VI'])
        self.no_samples_VI_test = int(cfg_trainer['no_samples_VI_test'])
        self.log_period_VI = cfg_trainer['log_period_VI']

        self.MCMC = cfg_trainer['MCMC']
        self.MCMC_init = cfg_trainer['MCMC_init']  # NOTE (DG): one of ['VI', 'identity', 'noise']
        self.no_chains = int(cfg_trainer['no_chains'])
        self.no_samples_MCMC = int(cfg_trainer['no_samples_MCMC'])
        self.no_iters_burn_in = int(cfg_trainer['no_iters_burn_in'])
        self.log_period_MCMC = cfg_trainer['log_period_MCMC']

        # metrics and prints
        self.writer = TensorboardWriter(config.log_dir, cfg_trainer['tensorboard'])
        self.writer.write_hparams(config.config_str)
        self.metrics = MetricTracker(*[m for m in metrics], writer=self.writer)

    @abstractmethod
    def _run_model(self):
        raise NotImplementedError

    def run(self):
        self._run_model()

