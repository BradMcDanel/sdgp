import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layer import weight_quantize_fn, act_quantization
from utils import make_pair, nongrad_param

PRUNE_TYPE_RANDOM = 0
PRUNE_TYPE_MAX = 1

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
        self.prune_type = nongrad_param(quant_params['prune_type'])

        self.weight_quant = weight_quantize_fn(w_bit=self.w_bits.item())
        self.act_alq = act_quantization(self.x_bits.item())
        self.act_alpha = nn.Parameter(torch.tensor(8.0))

        # normal conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.bias = None
        
        shape = out_channels, in_channels // groups, *self.kernel_size
        self.weight = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')


    def forward(self, x):
        # Perform quantization
        x = self.act_alq(x, self.act_alpha)
        w = self.weight_quant(self.weight)

        # Perform convolution with quantized tensors
        x = F.conv2d(x, w, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)


        # Perform GSR on gradients
        if self.g_bits.item() != -1:
            x = GSRBackward.apply(x, self.g_bits, self.g_nonzero,
                                  self.g_groupsize, self.g_groupdim,
                                  self.prune_type)


        return x


    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


if __name__ == '__main__':
    quant_params = {
        'w_bits': 5,
        'x_bits': 5,
        'g_bits': 8,
        'g_nonzero': 2,
        'g_groupsize': 4,
        'prune_type': PRUNE_TYPE_RANDOM,
    }

    model = GSRConv2d(quant_params, 4, 8, 3)
    x = torch.Tensor(8, 4, 32, 32).uniform_()
    out = model(x)
    model.zero_grad()
    out.backward(torch.randn(*out.shape))

    


