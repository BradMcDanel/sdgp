import sys
sys.path.insert(0,'..')
from functools import partial

from types import SimpleNamespace
import resnet

arch = partial(resnet.resnet18, num_models=1)

_cfg = {
    'data_folder': '/hdd1/datasets',
    'save_path': 'saved_models/cifar10_resnet18_1m.pth',
    'arch': arch,
    'workers': 4,
    'momentum': 0.9,
    'epochs': 150,
    'batch_size': 256,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'gpu': None,
    'verbose': False,
}

cfg = SimpleNamespace(**_cfg)