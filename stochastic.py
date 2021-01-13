import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

def make_pair(x):
    if type(x) == int or type(x) == float:
        return x, x
    
    return x

def param_wrap(x):
    return nn.Parameter(torch.Tensor([x]), requires_grad=False)

class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', stochastic_params=None):
        super(StochasticConv2d, self).__init__()

        if stochastic_params == None:
            stochastic_params = {
                'w_prune': 0.0,
                'w_group': 1,
                'w_stoch': False,
                'x_prune': 0.0,
                'x_group': 1,
                'x_stoch': False,
                'g_prune': 0.0,
                'g_group': 1,
                'g_stoch': False,
            }

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.w_prune = param_wrap(stochastic_params['w_prune'])
        self.w_group = param_wrap(stochastic_params['w_group'])
        self.w_stoch = param_wrap(1 if stochastic_params['w_stoch'] else 0)
        self.x_prune = param_wrap(stochastic_params['x_prune'])
        self.x_group = param_wrap(stochastic_params['x_group'])
        self.x_stoch = param_wrap(1 if stochastic_params['x_stoch'] else 0)
        self.g_prune = param_wrap(stochastic_params['g_prune'])
        self.g_group = param_wrap(stochastic_params['g_group'])
        self.g_stoch = param_wrap(1 if stochastic_params['g_stoch'] else 0)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.bias = None
        
        shape = out_channels, in_channels // groups, *self.kernel_size
        self.weight = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
    
    def forward(self, x):
        if self.x_prune > 0.0:
            x = StochasticForward.apply(x, self.x_prune, self.x_stoch,
                                        self.x_group)

        if self.w_prune > 0.0:
            w = StochasticForward.apply(self.weight, self.w_prune, self.w_stoch,
                                        self.w_group)
        else:
            w = self.weight

        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation,
                     self.groups)

        if self.g_prune > 0.0:
            x = StochasticBackward.apply(x, self.g_prune, self.g_stoch,
                                         self.g_group)

        return x

    def stochastic_weight(self):
        return StochasticForward.apply(self.weight, self.w_prune)

class StochasticForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prune_pct, is_stoch, group_size):
        x = x.clone()
        num_samples = 1024
        prune_pct = prune_pct.item()
        abs_x = x.abs()

        cut_pount = round(num_samples * prune_pct)
        tau = torch.kthvalue(abs_x.view(-1)[:num_samples], cut_pount)[0]

        # rounded idxs
        if is_stoch.item():
            r = torch.rand(x.size(), device=x.device)
        else:
            r = 0.5 * torch.ones_like(x, device=x.device)

        rounded_idxs = (abs_x < tau) & (abs_x > r*tau)
        x[rounded_idxs] = x.sign()[rounded_idxs] * tau

        # prune idxs
        prune_idxs = (abs_x < tau) & (abs_x <= r*tau)
        x[prune_idxs] = 0

        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class StochasticBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prune_pct, is_stoch, group_size):
        ctx.save_for_backward(prune_pct, is_stoch, group_size)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        num_samples = 1024
        prune_pct, is_stoch, group_size = ctx.saved_tensors
        prune_pct = prune_pct.item()
        group_size = int(group_size.item())
        abs_grad = grad_output.abs()

        B, C, W, H = abs_grad.shape
        abs_grad = abs_grad.view(B, -1, group_size, W, H)
        group_means = abs_grad.mean(2)
        grad_group = group_means.repeat_interleave(repeats=group_size, dim=1)
        grad_group = grad_group.view(B, C, W, H)

        cut_pount = round(num_samples * prune_pct)
        # tau = torch.kthvalue(group_means.view(-1)[:num_samples], cut_pount)[0]
        tau = torch.kthvalue(abs_grad.view(-1)[:num_samples], cut_pount)[0]

        # rounded idxs
        if is_stoch.item():
            r = torch.rand(group_means.size(), device=group_means.device)
        else:
            r = 0.5 * torch.ones_like(group_means, device=group_means.device)
        
        r = r.repeat_interleave(repeats=group_size, dim=1)
        r = r.view(B, C, W, H)
        
        rounded_idxs = (grad_group < tau) & (grad_group > r*tau)
        grad_output[rounded_idxs] = grad_output.sign()[rounded_idxs] * tau

        # prune idxs
        prune_idxs = grad_group <= r*tau
        grad_output[prune_idxs] = 0

        # vis = grad_output.detach()
        # plt.imshow(vis[:64, :64, 2, 2].cpu().numpy())
        # plt.colorbar()
        # plt.savefig('figures/group.png', dpi=300)
        # plt.clf()

        # plt.hist(grad_output.view(-1).detach().cpu().numpy(), bins=100)
        # plt.savefig('figures/hist2.png', dpi=300)
        # plt.clf()
        # assert False

        return grad_output, None, None, None
