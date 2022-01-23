import sys
sys.path.insert(0, '..')

from types import SimpleNamespace
import resnet
from functools import partial
import gsr

quant_params = {
    'w_bits': 32,
    'x_bits': 32,
    'g_bits': 32,
    'g_nonzero': 2,
    'g_groupsize': 4,
    'prune_type': gsr.PRUNE_TYPE_MAX,
}

arch = partial(resnet.gsr_resnet18, quant_params=quant_params)

_cfg = {
    'data_folder': '/data/datasets',
    'save_path': 'saved_models/cifar10_resnet18_bscale.pth',
    'arch': arch,
    'workers': 8,
    'momentum': 0.9,
    'epochs': 150,
    'batch_size': 512*4,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'gpu': None,
    'verbose': True,
}

cfg = SimpleNamespace(**_cfg)
