import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from stochastic import StochasticConv2d

class BasicBlockWSE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockWSE, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w_prune=0, x_prune=0, g_prune=0,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicStochasticBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stochastic_params, stride=1):
        super(BasicStochasticBlock, self).__init__()
        self.conv1 = StochasticConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False, stochastic_params=stochastic_params)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StochasticConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            stochastic_params=stochastic_params)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StochasticConv2d(
                    in_planes, self.expansion*planes, kernel_size=1,
                    stride=stride, bias=False, stochastic_params=stochastic_params),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StochasticResNet(nn.Module):
    def __init__(self, block, num_blocks, stochastic_params, num_classes=10):
        super(StochasticResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],
                                       stochastic_params, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],
                                       stochastic_params, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],
                                       stochastic_params, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],
                                       stochastic_params, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stochastic_params, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stochastic_params,
                                stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class ResNetWSE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_models=1):
        super(ResNetWSE, self).__init__()
        self.num_models = num_models
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        linear = []
        for _ in range(num_models):
            linear.append(nn.Linear(512*block.expansion, num_classes))
        self.linear = nn.ModuleList(linear)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        N = out.shape[1]
        for i in range(self.num_models):
            masked_out = out.clone()
            if self.num_models > 1:
                chunk_size = N // self.num_models
                masked_out[:, i*chunk_size:(i+1)*chunk_size] = 0

            if i == 0:
                final = self.linear[i](masked_out)
            else:
                final += self.linear[i](masked_out)

        return final

    def get_heads(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        outputs = []
        N = out.shape[1]
        for i in range(self.num_models):
            masked_out = out.clone()
            chunk_size = N // self.num_models
            masked_out[:, i*chunk_size:(i+1)*chunk_size] = 0
            outputs.append(self.linear[i](masked_out))
        
        return torch.stack(outputs)
    
    def masked_heads(self, x, num_masks=4):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        outputs = []
        N = out.shape[1]
        for i in range(num_masks):
            masked_out = torch.zeros_like(out)
            # masked_out = out.clone()
            chunk_size = N // num_masks
            s, e = i*chunk_size, (i+1)*chunk_size
            masked_out[:, s:e] = out[:, s:e]
            # print(i*chunk_size, (i+1)*chunk_size)
            outputs.append(self.linear[0](masked_out))
        
        return torch.stack(outputs)
    

def stochastic_resnet18(block=None, stochastic_params=None):
    if block == None:
        block = BasicBlock
    
    return StochasticResNet(block, [2, 2, 2, 2], stochastic_params)

def resnet18(block=None, num_models=1):
    if block == None:
        block = BasicBlock
    return ResNetWSE(block, [2, 2, 2, 2], num_models=num_models)

# def resnet50(block=None, num_models=1):
#     if block == None:
#         block = Bottleneck
#     return ResNet(block, [3, 4, 6, 3], num_models=num_models)

