import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from quant_layer import weight_quantize_fn, act_quantization
from utils import make_pair, nongrad_param

PRUNE_TYPE_MAX = 0


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
                                            conf['nonzero'], conf['groupsize'])
            input_grad = conv_back_input(input.shape, weight, prune_grad_output,
                                         conf["stride"], conf["padding"],
                                         conf["dilation"], conf["groups"],
                                         False, False, False)
        if ctx.needs_input_grad[1]:
            # pruning across batch dimension
            weight_grad = conv_back_weight(input, weight.shape, grad_output,
                                           conf["stride"], conf["padding"],
                                           conf["dilation"], conf["groups"],
                                           False, False, False)

        if bias is not None and ctx.needs_input_grad[2]:
            bias_grad = grad_output.sum(dim=(0, 2, 3))

        return input_grad, weight_grad, bias_grad, stride_grad, padding_grad, dilation_grad, groups_grad, gsr_params_grad


def SR(input, bins, max_offset=0.9):
    max_v = max_offset*input.abs().max()
    sf = max_v / bins
    y = (input / sf).abs()
    frac = y - y.floor()
    rnd = torch.rand(y.shape, device=y.device)
    j = rnd <= frac
    y[j] = y[j].ceil()
    y[~j] = y[~j].floor()
    y = y.clamp(-bins, bins)
    y = input.sign() * y

    return y * sf


def GSR(input, bins, num_nonzero, group_size, group_dim, prune_type):

    # compute group shapes
    shape = input.shape
    group_shape = list(input.unsqueeze(group_dim).shape)
    group_shape[group_dim] = group_shape[group_dim+1]//group_size
    group_shape[group_dim+1] = group_size
    
    # compute masks to set values to 0 based on num_nonzero
    input = input.view(*group_shape)

    if prune_type == PRUNE_TYPE_MAX:
        idx = input.abs().sort(group_dim+1)[0]
        idx = idx.index_select(group_dim + 1,
                               torch.tensor([num_nonzero - 1],
                                            device=input.device))
        idx = idx.repeat_interleave(group_size, group_dim+1)
        mask = input.abs() > idx

        # stochastic_scale = 1 # likely 1 is incorrect here
    elif prune_type == PRUNE_TYPE_RANDOM:
        rand_mat = torch.rand(input.shape, device=input.device)
        idx = rand_mat.sort(group_dim+1)[0]
        idx = idx.index_select(group_dim + 1,
                               torch.tensor([num_nonzero - 1],
                                            device=input.device))
        idx = idx.repeat_interleave(group_size, group_dim+1)
        mask = rand_mat <= idx

        # scale based on ratio of num_nonzero
        # stochastic_scale = (group_size / num_nonzero)
    else:
        raise ValueError("Invalid prune_type: {}. Options are: " \
                         "PRUNE_TYPE_RANDOM, PRUNE_TYPE_MAX")

    # set values to out.backward(torch.randn(1, 10))
    res = torch.zeros_like(input)
    res[mask] = input[mask]

    # apply stochastic scale due to pruning
    scale = input.abs().sum(group_dim + 1) / res.abs().sum(group_dim + 1)
    res = res * scale.unsqueeze(group_dim + 1)
    # res = res * stochastic_scale

    #stochastically round res
    # res = SR(res, bins)

    res = res.view(*shape)

    return res


class GSRBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, g_bits, num_nonzero, group_size, group_dim,
                prune_type):
        ctx.save_for_backward(g_bits, num_nonzero, group_size, group_dim,
                              prune_type)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        g_bits, num_nonzero, group_size, group_dim, prune_type = ctx.saved_tensors
        grad_output = GSR(grad_output, 2 ** (g_bits.item() - 1),
                          num_nonzero.item(), group_size.item(), group_dim.item(),
                          prune_type)
        
        return grad_output, None, None, None, None, None


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


if __name__ == '__main__':
    quant_params = {
        'nonzero': 2,
        'groupsize': 4,
        'prune_type': PRUNE_TYPE_RANDOM,
    }

    model = GSRConv2d(quant_params, 4, 8, 3)
    x = torch.Tensor(8, 4, 32, 32).uniform_()
    out = model(x)
    model.zero_grad()
    out.backward(torch.randn(*out.shape))

