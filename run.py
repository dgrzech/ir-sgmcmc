import argparse
from datetime import datetime

import numpy as np
import torch

from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(config):
    # data loader
    data_loader = config.init_data_loader()

    # losses
    losses = config.init_losses()
    
    # transformation and registration modules
    transformation_module, registration_module = config.init_transformation_and_registration_modules()

    # losses
    metrics = config.init_metrics()

    # run the model
    trainer = Trainer(config, data_loader, losses, transformation_module, registration_module, metrics)
    trainer.run()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='MCMC')

    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args = parser.parse_args()

    # config
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    config = ConfigParser.from_args(parser, timestamp=timestamp)

    # run the model
    run(config)
