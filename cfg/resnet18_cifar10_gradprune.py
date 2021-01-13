import sys
sys.path.insert(0, '..')

from types import SimpleNamespace
import resnet
import batchnorm
from functools import partial

# norm_layer = partial(batchnorm.GradBatchNorm2d, prune_pct=0.75)

_cfg = {
    'data_folder': '/hdd1/datasets',
    'save_path': 'saved_models/cifar10_resnet18_prune_75_stoc.pth',
    'arch': partial(resnet.resnet18, block=resnet.BasicBlockGradPrune),
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