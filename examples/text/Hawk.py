import aka.nn as nn
import aka.numpy as np

def HawkBlock(**kwargs):
    '''
    Paper: Griffin & Hawk
    Changes to paper:
        1, Add num_heads to RG_LRU. The orginal paper is element-wise RG_LRU (num_heads==D)
        2, Add silu after conv.
        3, beta = 1 - alpha. The orginal paper is beta = sqrt(1-alpha**2)
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.in_proj = nn.Linear(args.latent_dim, self.hidden_dim*2, bias=args.bias)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=args.bias,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim,
            padding=0,
        )
        self.num_heads = getattr(args, 'num_heads', 1)
        self.r_gate = nn.Linear(self.hidden_dim, self.num_heads)
        self.i_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
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
            if 'gru_state' in state:
                gru_state = state['gru_state']
                convx = np.cat((state['conv_state'], convx), dim=2)
            else:
                gru_state = np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, device=x.device)
            state['conv_state'] = convx[:, :, -n_conv_state:].detach()
        else:
            n_conv_state = 0
            gru_state = np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, device=x.device)

        if convx.size(2) < l + n_conv_state:
            convx = np.pad(convx, (l + n_conv_state - convx.size(2), 0), value=0.)
        x = self.conv1d(convx)
        x = np.silu(x)
        x = np.rearrange('b d l->b l d', x)

        mask = np.tril(np.ones(l,l,device=x.device))
        r = np.sigmoid(self.r_gate(x))
        r = np.exp((self.c * np.softplus(self.delta)) * r)          # [B,L,H]
        cuma = np.cumprod(r, dim=1)     # [a(1), a(1)*a(2), .... , a(1)*...*a(n)]
        shifta = np.pad(cuma, (0,0,1,-1), value=1.0)

        x = np.sigmoid(self.i_gate(x)) * x
        x = np.rearrange('b l (h d)->b l h d', x, h=self.num_heads) # [B,L,H,D]
        shiftx = np.cat([gru_state,x[:,:l-1]], dim=1) - x

        shiftb = shiftx / (1.0e-10 + shifta.unsqueeze(-1))    # 'blhd,blh->blhd'
        y = np.einsum('blh,lm,bmhd->blhd', cuma, mask, shiftb) + x
        if state is not None:
            state['gru_state'] = y[:,-1:].detach()
        y = np.rearrange('b l h d->b l (h d)',y)
        y = y * np.gelu(gate)
        return self.out_proj(y)
    return __init__(nn.Module(forward = forward),**kwargs)

def HawkArgs(name):
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [dict(
            name = 'Hawk',
            num_heads = 8,
            conv_kernel_size = 4
        )]*16,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match(name):
        case 'Hawk':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Hawk',
                        num_heads = 8,
                    ),
                    dict(
                        name = "MLP",
                        kv_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'HawkOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'Hawk',
                    num_heads = 8,
                )]*16,
            )
        case 'Griffin':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Attention',
                        num_heads = 8,
                        num_kv_groups = 8,
                        rotary_embedding = True,
                    ),
                    dict(
                        name = 'Hawk',
                        num_heads = 8,
                    ),
                ]*8,
            )
        case 'MambaOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'Mamba',
                    num_heads = 8,
                )]*16,
            )
        case 'SSMOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'SSM',
                    num_heads = 8,
                )]*16,
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'Hawk-Hawk',
        'Hawk-MambaOnly',
        'Hawk-SSMOnly',
        'Hawk-HawkOnly',
        # 'Hawk-Griffin',
    ]
    TrainRoles(roles, lr = 6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
