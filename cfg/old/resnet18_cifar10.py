import sys
sys.path.insert(0,'..')
from functools import partial

from types import SimpleNamespace
import resnet

arch = partial(resnet.resnet18)

_cfg = {
    'data_folder': '/data/datasets',
    'save_path': 'saved_models/cifar10_resnet18.pth',
    'arch': arch,
    'workers': 4,
    'momentum': 0.9,
    'epochs': 150,
    'batch_size': 512*4,
    'lr': 0.8,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'gpu': None,
    'verbose': True,
}

cfg = SimpleNamespace(**_cfg)
