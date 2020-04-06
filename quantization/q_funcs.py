import math
import numpy as np
import torch.nn as nn

def uniform_quantize(k):
# Refernce code from: https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/a12b6171555b8aba4d45e7dba6aeeab46be5480e/utils/quant_dorefa.py#L66
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

def get_q_functions(wbit, abit, method='dorefa'):
    """ Returns quantization functions for weights and activations of
    convolutions and linear blocks

    Inputs:
    - wbit: int for desired bit quantization
    - abit: int for desired activation bit quantization
    - method: string with following quantization methods:
            'asymRBL, symRBL, dorefa, pact, wrpn'
    """

    if method == 'dorefa':
        w_qfn, act_qfn = get_dorefa_fns(wbit, abit)

    return w_qfn, act_qfn


def get_dorefa_fns(wbit, abit):

    class dorefa_weight_quantize(nn.Module):
        def __init__(self, wbit):
            super(dorefa_weight_quantize, self).__init__()
            self.wbit = wbit
            self.uniform_q = uniform_quantize(k=wbit)

        def forward(self, x):
            # TODO: check wbit flags
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / (2 * max_w) + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)

            return weight_q

    class dorefa_activations_quantize(nn.Module):
        def __init__(self, abit):
            super(dorefa_activations_quantize, self).__init__()
            self.abit=abit
            self.uniform_q = uniform_quantize(k=abit)

        def forward(self, x):
            if self.a_bit == 32:
              activation_q = x
            else:
              activation_q = self.uniform_q(torch.clamp(x, 0, 1))
              # print(np.unique(activation_q.detach().numpy()))
            return activation_q

    return dorefa_weight_quantize, dorefa_activations_quantize