import sys
sys.path.insert(0, '..')

from types import SimpleNamespace
import resnet
from functools import partial

stochastic_params = {
    'w_prune': 0.0,
    'w_group': 1,
    'w_stoch': False,
    'x_prune': 0.0,
    'x_group': 1,
    'x_stoch': False,
    'g_prune': 0.8,
    'g_group': 8,
    'g_stoch': True,
}

arch = partial(resnet.stochastic_resnet18, block=resnet.BasicStochasticBlock,
               stochastic_params=stochastic_params)

_cfg = {
    'data_folder': '/hdd1/datasets',
    'save_path': 'saved_models/cifar10_resnet18_w00n_x00n_g80ygr.pth',
    'arch': arch,
    'workers': 4,
    'momentum': 0.9,
    'epochs': 150,
    'batch_size': 256,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'gpu': None,
    'verbose': True,
}

cfg = SimpleNamespace(**_cfg)