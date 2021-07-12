import sys
sys.path.insert(0, '..')

from types import SimpleNamespace
import resnet
from functools import partial

quant_params = {
    'w_bits': 4,
    'x_bits': 4,
    'g_bits': 4,
    'g_nonzero': 4,
    'g_groupsize': 4,
}

arch = partial(resnet.gsr_resnet18, quant_params=quant_params)

_cfg = {
    'data_folder': '/hdd1/datasets',
    'save_path': 'saved_models/cifar10_resnet18_4_4_4_1_4.pth',
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