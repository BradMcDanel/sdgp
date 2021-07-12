import torch
import torch.nn as nn
import torch.nn.functional as F

from lsq import LSQ
from utils import make_pair, nongrad_param

def SR(input, bins):
    max_v = input.abs().max()
    sf = max_v / bins
    y = (input / sf).abs()
    frac = y - y.floor()
    rnd = torch.rand(y.shape, device=y.device)
    j = rnd <= frac
    y[j] = y[j].ceil()
    y[~j] = y[~j].floor()
    y = input.sign() * y

    return y * sf

def SRG(input, bins, num_nonzero, group_size, group_dim):
    #stochastically round input
    roud = SR(input, bins)

    # compute group shapes
    shape = roud.shape
    group_shape = list(roud.unsqueeze(group_dim).shape)
    group_shape[group_dim] = group_shape[group_dim+1]//group_size
    group_shape[group_dim+1] = group_size
    
    # compute masks to set values to 0 based on num_nonzero
    roud = roud.view(*group_shape)
    rand_mat = torch.rand(roud.shape)
    idx = rand_mat.sort(group_dim+1)[0]
    idx = idx.index_select(group_dim+1, torch.tensor([num_nonzero-1]))
    idx = idx.repeat_interleave(group_size, group_dim+1)
    mask = rand_mat <= idx

    # set values to 0
    res = torch.zeros_like(roud)
    res[mask] = roud[mask]
    res = res.view(*shape)

    # scale based on ratio of num_nonzero
    res = res * (group_size / num_nonzero) 

    return res


class GSRBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, g_bits, num_nonzero, group_size, group_dim):
        ctx.save_for_backward(g_bits, num_nonzero, group_size, group_dim)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        g_bits, num_nonzero, group_size, group_dim = ctx.saved_tensors
        grad_output = SRG(grad_output, 2 ** (g_bits.item() - 1),
                          num_nonzero.item(), group_size.item(), group_dim.item())
        
        return grad_output, None, None, None, None


class GSRConv2d(nn.Module):
    def __init__(self, quant_params, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(GSRConv2d, self).__init__()

        # unpack quant params
        self.w_bits = nongrad_param(quant_params['w_bits'])
        self.x_bits = nongrad_param(quant_params['x_bits'])
        self.g_bits = nongrad_param(quant_params['g_bits'])
        self.g_groupsize = nongrad_param(quant_params['g_groupsize'])
        self.g_nonzero = nongrad_param(quant_params['g_nonzero'])
        self.g_groupdim = nongrad_param(1)

        # normal conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.x_sf = nn.Parameter(torch.Tensor([1]))
        self.w_sf = nn.Parameter(torch.Tensor([1]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.bias = None
        
        shape = out_channels, in_channels // groups, *self.kernel_size
        self.weight = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # Perform quantization
        x = LSQ.apply(x, self.x_sf, self.x_bits)
        w = LSQ.apply(self.weight, self.w_sf, self.w_bits)

        # Perform convolution with quantized tensors
        x = F.conv2d(x, w, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)


    	# Perform GSR on gradients
        x = GSRBackward.apply(x, self.g_bits, self.g_nonzero,
                              self.g_groupsize, self.g_groupdim)


        return x
