import json
import logging
from functools import reduce
from operator import getitem
from pathlib import Path

import numpy as np

import data_loader.data_loaders as module_data
import model.distributions as model_distr
import model.loss as model_loss
import utils.registration as registration
import utils.transformation as transformation
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, modification=None, timestamp=None):
        # load config file and apply modification
        self._config = _update_config(config, modification)

        # logger
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

        verbosity = self['trainer']['verbosity']
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity

        self._logger = logging.getLogger('default')
        self._logger.setLevel(self.log_levels[verbosity])

        # set save_dir where trained model and log will be saved.
        run_id = timestamp

        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']
        dir = save_dir / exper_name / run_id

        self._dir = dir
        self._log_dir = dir / 'log'
        self._save_dir = dir / 'models'
        self._tensors_dir = dir / 'tensors'
        self._samples_dir = dir / 'samples'

        self._im_dir = dir / 'images'
        self._fields_dir = dir / 'fields'
        self._grids_dir = dir / 'grids'
        self._norms_dir = dir / 'norms'
        
        # segmentation IDs
        self.structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12,
                                'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17,
                                'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49,
                                'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52,
                                'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}

        # make directories for saving checkpoints and log.
        exist_ok = run_id == ''
        
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.tensors_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.samples_dir.mkdir(parents=True, exist_ok=exist_ok)

        samples_VI_dir = self.samples_dir / 'VI'
        samples_MCMC_dir = self.samples_dir / 'MCMC'

        samples_VI_dir.mkdir(parents=True, exist_ok=exist_ok)
        samples_MCMC_dir.mkdir(parents=True, exist_ok=exist_ok)

        self.im_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.fields_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.grids_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.norms_dir.mkdir(parents=True, exist_ok=exist_ok)
        
        # configure logging 
        setup_logging(self.log_dir)

        # save updated config file to the checkpoint dir
        self.config_str = json.dumps(self.config, indent=4, sort_keys=False).replace('\n', '')
        write_json(self.config, dir / 'config.json')

    @classmethod
    def from_args(cls, args, options='', timestamp=None):
        # initialise this class from cli arguments
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}

        return cls(config, modification=modification, timestamp=timestamp)

    def init_data_loader(self):
        self['data_loader']['args']['save_dirs'] = self.save_dirs
        return self.init_obj('data_loader', module_data)

    def init_losses(self):
        data_loss = self.init_obj('data_loss', model_loss)
        data_loss_scale_prior = self.init_obj('data_loss_scale_prior', model_distr)
        data_loss_proportion_prior = self.init_obj('data_loss_proportion_prior', model_distr)

        entropy_loss = self.init_obj('entropy_loss', model_loss)

        self['reg_loss']['args']['dims'] = self['data_loader']['args']['dims']
        reg_loss = self.init_obj('reg_loss', model_loss)

        losses_dict = {'data':
                           {'loss': data_loss, 'scale_prior': data_loss_scale_prior, 'proportion_prior': data_loss_proportion_prior},
                       'reg': {'loss': reg_loss}, 'entropy': entropy_loss}

        if reg_loss.__class__.__name__ == 'RegLoss_LogNormal':
            self['reg_loss_loc_prior']['args']['dof'] = np.prod(self['data_loader']['args']['dims']) * 3.0
            reg_loss_loc_prior = self.init_obj('reg_loss_loc_prior', model_distr)
            reg_loss_scale_prior = self.init_obj('reg_loss_scale_prior', model_distr)

            losses_dict['reg']['loc_prior'] = reg_loss_loc_prior
            losses_dict['reg']['scale_prior'] = reg_loss_scale_prior

        return losses_dict

    def init_metrics(self):
        cfg_data_loss = self['data_loss']['args']
        no_components = cfg_data_loss['no_components']

        # model parameters
        scales_VI = ['VI/train/GMM/scale_' + str(idx) for idx in range(no_components)] + ['VI/train/reg/scale']
        scales_MCMC = ['MCMC/GMM/scale_' + str(idx) for idx in range(no_components)] + ['MCMC/reg/scale']

        proportions_VI = ['VI/train/GMM/proportion_' + str(idx) for idx in range(no_components)]
        proportions_MCMC = ['MCMC/GMM/proportion_' + str(idx) for idx in range(no_components)]

        locs_VI = ['VI/train/reg/loc']
        locs_MCMC = ['MCMC/reg/loc']

        VD_VI = ['VI/train/VD/alpha']
        VD_MCMC = ['MCMC/VD/alpha']

        model_params = locs_VI + locs_MCMC + scales_VI + scales_MCMC + proportions_VI + proportions_MCMC + VD_VI + VD_MCMC

        # losses
        loss_terms_VI = ['VI/train/data_term', 'VI/train/reg_term', 'VI/train/entropy_term', 'VI/train/total_loss']
        loss_terms_MCMC = ['MCMC/data_term', 'MCMC/reg_term', 'MCMC/total_loss']

        losses = loss_terms_VI + loss_terms_MCMC

        # other
        other = ['VI/train/reg/energy'] + ['MCMC/reg/energy'] + ['VI/train/max_updates/' + parameter for parameter in ['mu', 'log_var', 'u']]

        # metrics
        modes = ['train', 'test']
        
        ASD_VI = ['VI/' + mode + '/ASD/' + structure for structure in self.structures_dict for mode in modes]
        DSC_VI = ['VI/' + mode + '/DSC/' + structure for structure in self.structures_dict for mode in modes]

        ASD_MCMC = ['MCMC/ASD/' + structure for structure in self.structures_dict]
        DSC_MCMC = ['MCMC/DSC/' + structure for structure in self.structures_dict]
        
        no_non_diffeomorphic_voxels_VI = ['VI/' + mode + '/no_non_diffeomorphic_voxels' for mode in modes]
        no_non_diffeomorphic_voxels_MCMC = ['MCMC/no_non_diffeomorphic_voxels']

        metrics = ASD_VI + ASD_MCMC + DSC_VI + DSC_MCMC + \
                  no_non_diffeomorphic_voxels_VI + no_non_diffeomorphic_voxels_MCMC

        return model_params + losses + metrics + other

    def init_transformation_and_registration_modules(self):
        self['transformation_module']['args']['dims'] = self['data_loader']['args']['dims']

        return self.init_obj('transformation_module', transformation), \
               self.init_obj('registration_module', registration)

    def init_obj(self, name, module, *args, **kwargs):
        """
        find a function handle with the name given as 'type' in config, and return the
        instance initialized with corresponding arguments given;
        `object = config.init_obj('name', module, a, b=1)` is equivalent to `object = module.name(a, b=1)`
        """

        module_name = self[name]['type']

        if 'args' in dict(self[name]):
            module_args = dict(self[name]['args'])
            module_args.update(kwargs)
        else:
            module_args = dict()

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        # access items like in a dict
        return self.config[name]

    # setting read-only attributes
    @property
    def logger(self):
        return self._logger

    @property
    def config(self):
        return self._config

    @property
    def dir(self):
        return self._dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def tensors_dir(self):
        return self._tensors_dir

    @property
    def samples_dir(self):
        return self._samples_dir

    @property
    def im_dir(self):
        return self._im_dir

    @property
    def fields_dir(self):
        return self._fields_dir

    @property
    def grids_dir(self):
        return self._grids_dir

    @property
    def norms_dir(self):
        return self._norms_dir

    @property
    def save_dirs(self):
        return {'dir': self.dir, 'tensors': self.tensors_dir, 'samples': self.samples_dir,
                'images': self.im_dir, 'fields': self.fields_dir, 'grids': self.grids_dir, 'norms': self.norms_dir}


def _update_config(config, modification):
    # helper functions to update config dict with custom cli options
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    # set a value in a nested object in tree by sequence of keys
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    # access a nested object in tree by sequence of keys
    return reduce(getitem, keys, tree)

