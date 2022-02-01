import torch
from torch.autograd import Function
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_fwd, custom_bwd
import matplotlib.pyplot as plt

from utils import make_pair, nongrad_param

PRUNE_TYPE_MAX = 0
PRUNE_TYPE_RND = 1
PRUNE_TYPE_STC = 2
PRUNE_TYPE_MAX_NORM = 3

FP16_EPS = 1e-4

cudnn_convolution = load(name="cudnn_convolution",
                         sources=["cudnn_convolution.cpp"], verbose=False)
conv_fwd = cudnn_convolution.convolution
conv_back_input = cudnn_convolution.convolution_backward_input
conv_back_weight = cudnn_convolution.convolution_backward_weight

prune = load(name="prune", sources=["kernels/prune.cpp", "kernels/prune_kernel.cu"],
             verbose=False)

# convert str prune_type to int
def prune_type_to_int(prune_type):
    if prune_type == 'max':
        return PRUNE_TYPE_MAX
    elif prune_type == 'rnd':
        return PRUNE_TYPE_RND
    elif prune_type == 'stc':
        return PRUNE_TYPE_STC
    elif prune_type == 'maxnorm':
        return PRUNE_TYPE_MAX_NORM
    else:
        raise ValueError('prune_type must be "max", "rnd", "stc", or "maxnorm"')


def convert_model(model, prune_type, nonzero, group_size):
    for child_name, c in model.named_children():
        if isinstance(c, nn.Conv2d) and c.in_channels > 3:
            gsr_conv = GSRConv2d(prune_type, nonzero, group_size, c.in_channels,
                                 c.out_channels, c.kernel_size, c.stride,
                                 c.padding, c.dilation, c.groups, c.bias)
            setattr(model, child_name, gsr_conv)
        else:
            convert_model(c, prune_type, nonzero, group_size)
    
    return model


def prune_wrapper(x, prune_type, nonzero, groupsize):
    if prune_type == PRUNE_TYPE_RND or prune_type == PRUNE_TYPE_MAX:
        return prune.prune(x, prune_type, nonzero, groupsize)
    elif prune_type == PRUNE_TYPE_MAX_NORM:
        y = prune.prune(x, PRUNE_TYPE_MAX, nonzero, groupsize)
        x_sum = torch.abs(x).sum(dim=(2, 3), keepdim=True)
        y_sum = torch.abs(y).sum(dim=(2, 3), keepdim=True) + FP16_EPS
        ratio = x_sum / y_sum
        y = y * ratio
        y_sum = torch.abs(y).sum(dim=(2, 3), keepdim=True)
        return y
    else:
        raise ValueError('prune_type must be "max" or "maxnorm"')
        

class GSRConv2dFunc(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                prune_type, nonzero, group_size):
        ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "prune_type": prune_type,
            "nonzero": nonzero,
            "groupsize": group_size,
        }
        return conv_fwd(input, weight, bias, stride, padding, dilation, groups,
                        False, False)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        conf = ctx.conf
        input_grad = weight_grad = bias_grad = stride_grad = padding_grad = dilation_grad = groups_grad = gsr_prune_type_grad = gsr_nonzero = gsr_groupsize = None
        if ctx.needs_input_grad[0]:
            # pruning activation gradients
            prune_grad_output = prune_wrapper(grad_output, conf['prune_type'],
                                              conf['nonzero'], conf['groupsize'])
            input_grad = conv_back_input(input.shape, weight, prune_grad_output,
                                         conf["stride"], conf["padding"],
                                         conf["dilation"], conf["groups"],
                                         False, False, False)
        if ctx.needs_input_grad[1]:
            weight_grad = conv_back_weight(input, weight.shape, grad_output,
                                           conf["stride"], conf["padding"],
                                           conf["dilation"], conf["groups"],
                                           False, False, False)

        if bias is not None and ctx.needs_input_grad[2]:
            bias_grad = grad_output.sum(dim=(0, 2, 3))

        return input_grad, weight_grad, bias_grad, stride_grad, padding_grad, dilation_grad, groups_grad, gsr_prune_type_grad, gsr_nonzero, gsr_groupsize


class GSRConv2d(nn.Module):
    def __init__(self, prune_type, nonzero, group_size, in_channels,
                 out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(GSRConv2d, self).__init__()

        # gsr params
        self.prune_type = nongrad_param(prune_type_to_int(prune_type))
        self.nonzero = nongrad_param(nonzero)
        self.group_size = nongrad_param(group_size)

        # normal conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = make_pair(stride)
        self.padding = make_pair(padding)
        self.dilation = make_pair(dilation)
        self.groups = groups

        # initialize weight
        shape = out_channels, in_channels // groups, *self.kernel_size
        self.weight = nn.Parameter(torch.zeros(*shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels),
                                     requires_grad=False)

        if padding_mode != 'zeros':
            raise ValueError("Only zero padding is supported.")


    def forward(self, x):
        x = GSRConv2dFunc.apply(x, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups,
                                self.prune_type, self.nonzero, self.group_size)

        return x

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
                'padding={}, dilation={}, groups={}'.format(
                    self.in_channels, self.out_channels, self.kernel_size,
                    self.stride, self.padding, self.dilation, self.groups)


if __name__ == '__main__':
    import torchvision
    resnet18 = torchvision.models.resnet18(pretrained=False)
    model = convert_model(resnet18, PRUNE_TYPE_MAX, 2, 4)
