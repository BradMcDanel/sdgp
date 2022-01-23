import sys
sys.path.insert(0, '..')

from types import SimpleNamespace
import resnet
from functools import partial
import gsr

gsr_params = {
    'nonzero': 2,
    'groupsize': 4,
    'prune_type': gsr.PRUNE_TYPE_MAX,
}

arch = partial(resnet.gsr_resnet18, gsr_params=gsr_params)
NUM_GPUS = 4

_cfg = {
    'data_folder': '/data/datasets',
    'save_path': 'saved_models/cifar10_resnet18_gsr2_4_max.pth',
    'arch': arch,
    'workers': 8,
    'momentum': 0.9,
    'epochs': 150,
    'batch_size': 512*NUM_GPUS,
    'lr': 0.2*NUM_GPUS,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'gpu': None,
    'verbose': True,
}

cfg = SimpleNamespace(**_cfg)
