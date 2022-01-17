import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from utils import make_pair, nongrad_param

PRUNE_DIM_BATCH = 0
PRUNE_DIM_CHANNEL = 1

PRUNE_TYPE_MAX = 0
PRUNE_TYPE_RND = 1



cudnn_convolution = load(name="cudnn_convolution",
                         sources=["cudnn_convolution.cpp"], verbose=False)
conv_fwd = cudnn_convolution.convolution
conv_back_input = cudnn_convolution.convolution_backward_input
conv_back_weight = cudnn_convolution.convolution_backward_weight

prune = load(name="prune", sources=["kernels/prune.cpp", "kernels/prune_kernel.cu"],
             verbose=False)


class GSRConv2dFunc(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                gsr_params):
        ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "prune_type": gsr_params['prune_type'],
            "nonzero": gsr_params['nonzero'],
            "groupsize": gsr_params['groupsize']
        }
        return conv_fwd(input, weight, bias, stride, padding, dilation, groups,
                        False, False)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        conf = ctx.conf
        input_grad = weight_grad = bias_grad = stride_grad = padding_grad = dilation_grad = groups_grad = gsr_params_grad = None
        if ctx.needs_input_grad[0]:
            # pruning across channel dimension
            prune_grad_output = prune.prune(grad_output, conf['prune_type'],
                                            PRUNE_DIM_CHANNEL, conf['nonzero'],
                                            conf['groupsize'])
            input_grad = conv_back_input(input.shape, weight, prune_grad_output,
                                         conf["stride"], conf["padding"],
                                         conf["dilation"], conf["groups"],
                                         False, False, False)
        if ctx.needs_input_grad[1]:
            # pruning across batch dimension
            # prune_grad_output = prune.prune(grad_output, conf['prune_type'], 
            #                                 PRUNE_DIM_BATCH, conf['nonzero'],
            #                                 conf['groupsize'])
            weight_grad = conv_back_weight(input, weight.shape, grad_output,
                                           conf["stride"], conf["padding"],
                                           conf["dilation"], conf["groups"],
                                           False, False, False)

        if bias is not None and ctx.needs_input_grad[2]:
            bias_grad = grad_output.sum(dim=(0, 2, 3))

        return input_grad, weight_grad, bias_grad, stride_grad, padding_grad, dilation_grad, groups_grad, gsr_params_grad


class GSRConv2d(nn.Module):
    def __init__(self, gsr_params, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(GSRConv2d, self).__init__()

        # unpack gsr params
        self.gsr_params = gsr_params
        self.g_groupsize = nongrad_param(gsr_params['groupsize'])
        self.g_nonzero = nongrad_param(gsr_params['nonzero'])
        self.prune_type = nongrad_param(gsr_params['prune_type'])

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
        self.weight = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_(),
                                     requires_grad=False)

        if padding_mode != 'zeros':
            raise ValueError("Only zero padding is supported.")


    def forward(self, x):
        x = GSRConv2dFunc.apply(x, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups,
                                self.gsr_params)

        return x
