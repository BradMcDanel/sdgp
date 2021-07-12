import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def uniform_quant(x, bins):
    xdiv = x.mul((2 ** bins - 1))
    xhard = xdiv.round().div(2 ** bins - 1)
    return xhard


class LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, bins):
        input = input / alpha
        input_c = input.clamp(min=-1, max=1)
        sign = input_c.sign()
        input_abs = input_c.abs()
        input_q = uniform_quant(input_abs, bins).mul(sign)
        ctx.save_for_backward(input, input_q)
        input_q = input_q.mul(alpha)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input, input_q = ctx.saved_tensors
        i = (input.abs() > 1.0).float()
        sign = input.sign()
        grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum().view(1)
        return grad_input, grad_alpha, None
