import aka.nn as nn
import aka.numpy as np

def Conv1dBlock(**kwargs):
    def __init__(self, latent_dim, conv_kernel_size=4, bias=True, **kwargs):
        self.kernel_size = conv_kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            bias=bias,
            kernel_size=conv_kernel_size,
            groups=latent_dim,
            padding=0
        )
        return self

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        x = np.rearrange('b l d->b d l', x)
        if state is not None:
            conv_state = state.get('conv_state',None)
            if conv_state is not None:
                x = np.cat((conv_state, x), dim=2)
            state['conv_state'] = x[:, :, (1 - self.kernel_size):].detach()
        if x.size(2) < l + self.kernel_size - 1:
            x = np.pad(x, (l + self.kernel_size - 1 - x.size(2), 0), mode='replicate')
        x = self.conv1d(x)
        return np.rearrange('b d l->b l d', x)
    return __init__(nn.Module(forward = forward), **kwargs)
