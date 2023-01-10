import torch
import torch.nn as nn


class Hack_no_grad(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        with torch.no_grad():
            return self.module(*inputs, **kwargs)
