import aka.nn as nn
import aka.numpy as np

def CurvefitBlock(**kwargs):
    def __init__(self, n, cache_key='curve_cache'):
        self.n = n
        self.cache_key = cache_key
        match n:
            case 2:
                self.conv_kernel = np.array([[[-1.0, 2.0]]])
            case 3:
                self.conv_kernel = np.array([[[1.0, -3.0, 3.0]]])
            case 4:
                self.conv_kernel = np.array([[[-1.0, 3.0, -5.0, 4.0]]])
            case _:
                assert False
        return self

    def forward(self, inputs, state = None):
        (B, L, _) = inputs.shape

        x = inputs.view(B,L,-1)
        # -- append cache --
        if state is not None:
            cache_v = state.get(self.cache_key,None)
            if cache_v is not None:
                x = np.cat([cache_v,x], dim=1)
            state[self.cache_key] = x[:,1-self.n:].detach()

        # -- pad --
        if x.size(1) - L < self.n - 1:
            n_pad = self.n-1 + L - x.size(1)
            x = np.pad(x, (0, 0, n_pad, 0), mode='replicate')

        # -- conv --
        k = np.repeat(self.conv_kernel, x.size(2), dim=0)
        x = np.rearrange('b l d->b d l', x)
        x = np.conv1d(x, k, groups=x.size(1))
        x = np.rearrange('b d l->b l d', x)

        return x.view(inputs.shape)

    return __init__(nn.Module(forward=forward), **kwargs)

