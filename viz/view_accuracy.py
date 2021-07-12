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


baseline = torch.load('../saved_models/cifar10_resnet18_1m.pth')
gsr_a = torch.load('../saved_models/cifar10_resnet18_4_4_4_1_4.pth')
# prune50 = torch.load('saved_models/cifar10_resnet18_prune_50.pth')
# prune75 = torch.load('saved_models/cifar10_resnet18_stoc_50.pth')
# prune75stoc = torch.load('saved_models/cifar10_resnet18_prune_75_stoc.pth')

plt.plot(baseline['accs'], '-', color='k', linewidth=2, label='Baseline')
plt.plot(gsr_a['accs'], '-', color='r', linewidth=2, label='GSR(4/4/4/1/4)')
# plt.plot(baseline['accs'], '-', color='k', linewidth=2, label='Baseline')
# plt.plot(prune50['accs'], '-', color='r', linewidth=2, label ='50% gradients pruned')
# plt.plot(prune75['accs'], '-', color='b', linewidth=2, label ='80% Stochastic Pruning (w/d/g)')
# plt.plot(prune75stoc['accs'], '-', color='g', linewidth=2, label ='75% gradients pruned stoch')
plt.title('ResNet-18 CIFAR-10')
plt.title('ResNet-18 CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Test Classification Accuracy')

plt.legend(loc=0)
plt.show()
