import aka.nn as nn
import aka.numpy as np

def HawkBlock(args):
    '''
    Paper: Graffin & Hawk
    Changes to paper:
        1, Add num_heads to gater than paper. The orginal paper is element-wise RG_GRU (num_heads==D)
        2, Add silu after conv.
        3, beta = 1 - alpha. The orginal paper is beta = sqrt(1-alpha**2)
    '''
    def __init__(self, args):
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.in_proj = nn.Linear(args.latent_dim, self.hidden_dim*2, bias=args.bias)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=args.bias,
            kernel_size=self.conv_kernel_size,
            # groups=args.hidden_dim, # ？？？？？？？？？？？？
            padding=0,
        )
        self.num_heads = getattr(args, 'num_heads', 1)
        self.r_gate = nn.Linear(self.hidden_dim, self.num_heads, bias=args.bias)
        self.i_gate = nn.Linear(self.hidden_dim, self.hidden_dim, bias=args.bias)
        self.c = nn.Parameter(np.array(-8.), requires_grad=False)
        self.delta = nn.Parameter(np.array(0.5))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        convx = np.rearrange('b l d->b d l',x)
        if state is not None:
            n_conv_state = self.conv_kernel_size-1
            if 'conv_state' in state:
                gru_state = state['gru_state']
                convx = np.cat((state['conv_state'], convx), dim=2)
            else:
                gru_state = np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, device=x.device)
        else:
            n_conv_state = 0
            gru_state = np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, device=x.device)

        if convx.size(2) < l + n_conv_state:
            convx = np.pad(convx, (l + n_conv_state - convx.size(2), 0), value=0.)
        x = self.conv1d(convx)
        x = np.silu(x)
        x = np.rearrange('b d l->b l d',x)

        r = np.sigmoid(self.r_gate(x))
        a = np.exp((self.c * np.softplus(self.delta)) * r)    # [B,L,H]

        mask = np.tril(np.ones(l,l,device=x.device)).unsqueeze(-1)  # [  L,L,1]
        upA = a.unsqueeze(2)                        # [B,L,H]   -> [B,L,1,H]
        upA = np.where(mask==0, 1., upA)            # [B,L,1,H] -> [B,L,L,H]
        upA = np.cumprod(upA, dim=1) * mask

        ix = np.sigmoid(self.i_gate(x)) * x
        ix = np.rearrange('b l (h d)->b l h d', ix, h=self.num_heads)
        bx = (1.0-a.unsqueeze(-1)) * ix # np.sqrt(1. - a.unsqueeze(-1)**2) * ix
        h_and_b = np.cat([gru_state, bx[:,:l-1]], dim=1)    # [B,L,H,D]

        # What's the diff between the two lines below?
        # y = np.sum(upA.unsqueeze(-1)*abx.unsqueeze(2), dim=2) + bx
        y = np.einsum('blmh,blhd->blhd', upA, h_and_b) + bx
        if state is not None:
            state['conv_state'] = convx[:, :, -n_conv_state:].detach()
            state['gru_state'] = y[:,-1:].detach()
        y = np.rearrange('b l h d->b l (h d)',y)
        y = y * np.gelu(gate)
        return self.out_proj(y)
    return __init__(nn.Module(forward = forward),args)

