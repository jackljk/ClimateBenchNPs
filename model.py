import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    """Applies a module to each temporal slice of an input."""
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Reshape input to (batch * sequence, C, H, W)
        x_reshaped = x.contiguous().view(-1, *x.size()[2:]) if self.batch_first else x.contiguous().view(-1, *x.size()[1:])
        y = self.module(x_reshaped)

        # Reshape y back to (batch, sequence, C, H, W)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, *y.size()[1:])
        else:
            y = y.contiguous().view(-1, x.size(1), *y.size()[1:])
        return y