# From: https://github.com/yhhhli/APoT_Quantization/blob/master/CIFAR10/models/quant_layer.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.wgt_alpha = nn.Parameter(torch.tensor(3.0))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)      # weights normalization
            weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(x_bit):

    def uniform_quant(x, x_bit=3):
        xdiv = x.mul(2 ** x_bit - 1)
        xhard = xdiv.round().div(2 ** x_bit - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            input_q = uniform_quant(input_c, x_bit)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)