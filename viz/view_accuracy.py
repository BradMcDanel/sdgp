import sys
sys.path.insert(0, '..')

import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 14
TICK_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TICK_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=TICK_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure titl


import torch


baseline = torch.load('../saved_models/cifar10_resnet18.pth')
# quant = torch.load('../saved_models/cifar10_resnet18_5_5_-1_-1_-1.pth')
# sr = torch.load('../saved_models/cifar10_resnet18_5_5_5_8_8.pth')
sr2 = torch.load('../saved_models/cifar10_resnet18_5_5_8_1_1.pth')
gsr_2_4 = torch.load('../saved_models/cifar10_resnet18_5_5_8_2_4.pth')
gsr_2_4_noscale = torch.load('../saved_models/cifar10_resnet18_5_5_8_2_4_noscale.pth')
gsr_2_4_max = torch.load('../saved_models/cifar10_resnet18_5_5_8_2_4_max.pth')

# prune50 = torch.load('saved_models/cifar10_resnet18_prune_50.pth')
# prune75 = torch.load('saved_models/cifar10_resnet18_stoc_50.pth')
# prune75stoc = torch.load('saved_models/cifar10_resnet18_prune_75_stoc.pth')

plt.plot(baseline['accs'], '-', color='firebrick', linewidth=2, label='Dense')
# plt.plot(quant['accs'], '-', color='g', linewidth=2, label='Forward Quant (5/5/32)')
# plt.plot(sr['accs'], '-', color='r', linewidth=2, label='Forward+Backward Quant (5/5/5)')
# plt.plot(sr2['accs'], '-', color='orange', linewidth=2, label='Forward+Backward Quant (5/5/8)')
plt.plot(gsr_2_4['accs'], '-', color='forestgreen', linewidth=2, label='GSP (2:4)')
plt.plot(gsr_2_4_noscale['accs'], '-', color='royalblue', linewidth=2, label='Max (2:4)')
# plt.plot(baseline['accs'], '-', color='k', linewidth=2, label='Baseline')
# plt.plot(prune50['accs'], '-', color='r', linewidth=2, label ='50% gradients pruned')
# plt.plot(prune75['accs'], '-', color='b', linewidth=2, label ='80% Stochastic Pruning (w/d/g)')
# plt.plot(prune75stoc['accs'], '-', color='g', linewidth=2, label ='75% gradients pruned stoch')
plt.title('ResNet-18 on CIFAR-10')
plt.xlabel('Training Epoch')
plt.ylabel('Test Classification Accuracy (%)')

plt.legend(loc=0)
plt.savefig('../figures/cifar-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
