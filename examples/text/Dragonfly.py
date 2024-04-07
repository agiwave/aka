import aka.nn as nn
import aka.numpy as np

def DragonflyBlock(**kwargs):
    '''
    Args:
        latent_dim:       (required)
        hidden_dim:       latent_dim (default)
        num_heads:        8 (default)
        num_decay_groups: 1 (default)
        conv_kernel_size: 4 (default)
        prev_conv:        True(default)
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        bias = getattr(args, 'bias', False)
        dropout = getattr(args, 'dropout', 0.2)

        self.num_heads = getattr(args, 'num_heads', 8)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.vg_dim = self.hidden_dim if getattr(args, 'v_gate', False) else 0
        self.og_dim = args.latent_dim if getattr(args, 'o_gate', True) else 0
        self.num_decay_groups = getattr(args, 'num_decay_groups', 1)
        assert self.num_decay_groups <= (self.hidden_dim // self.num_heads)
        assert (self.hidden_dim // self.num_heads) % self.num_decay_groups == 0
        assert self.hidden_dim % self.num_heads == 0
        assert args.latent_dim % self.num_heads == 0

        # rg, ig, v, vg, og
        self.in_proj = nn.Parameter(shape=(
            self.num_heads,
            args.latent_dim//self.num_heads, 
            2 * self.num_decay_groups + self.hidden_dim//self.num_heads + self.vg_dim//self.num_heads + self.og_dim//self.num_heads))

        # o
        self.out_proj = nn.Parameter(shape=(
            self.num_heads,
            args.latent_dim // self.num_heads,
            self.hidden_dim // self.num_heads
        ))

        # self.in_proj = nn.Linear(args.latent_dim, self.num_heads*2 + self.hidden_dim + self.vg_dim + self.og_dim, bias=args.bias)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.prev_conv = getattr(args, 'prev_conv', True)
        self.post_conv = getattr(args, 'post_conv', False)
        self.conv1d = None if not (self.prev_conv or self.post_conv) else nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=args.bias,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim,
            padding=0
        )
        self.dropout_v = nn.Dropout(dropout)
        self.c = nn.Parameter(np.array(-8.), requires_grad=False)
        self.delta = nn.Parameter(np.array(0.5))
        return self

    def conv(x, k, kernel_size, state, key):
        (b, l, h, d) = x.shape
        x = np.rearrange('b l h d->b (h d) l', x)
        if state is not None:
            conv_state = state.get(key,None)
            if conv_state is not None:
                x = np.cat((state[key], x), dim=2)
            state[key] = x[:, :, (1 - kernel_size):].detach()
        if x.size(2) < l + kernel_size - 1:
            x = np.pad(x, (l + kernel_size - 1 - x.size(2), 0), mode='replicate')
        x = k(x)
        x = np.silu(x)
        return np.rearrange('b (h d) l->b l h d', x, h=h)

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        x = np.rearrange('b l (h d)->b l h d', x, h=self.num_heads)
        splits = np.einsum('b l h d, h d v->b l h v', x, self.in_proj)

        (rg, ig, x, vg, og) = splits.split([self.num_decay_groups, self.num_decay_groups, self.hidden_dim//self.num_heads, self.vg_dim//self.num_heads, self.og_dim//self.num_heads], dim=-1)
        
        # -- Prev Conv -- 
        x = x if not self.prev_conv else conv(x, self.conv1d, self.conv_kernel_size, state, 'prev_conv_state')

        # -- RG_LRU or GRU --
        rg = np.sigmoid(rg)
        rg = ((self.c * np.softplus(self.delta)) * rg)  # [B,L,H,D]

        replicate_num_groups = 1 if self.num_decay_groups == 1 else self.hidden_dim // self.num_heads // self.num_decay_groups
        if replicate_num_groups > 1:
            igg = np.repeat(ig, replicate_num_groups, dim=-1)
            irg = np.repeat(rg, replicate_num_groups, dim=-1)
        else:
            igg = ig
            irg = rg
        x = (1-np.exp(irg)) * np.sigmoid(igg) * x # The orginal paper: np.sqrt(1-rg**2)*np.sigmoid(ig).unsqueeze(-1) * x
        gru_state = None if state is None else state.get('gru_state',None)
        gru_state = gru_state if gru_state is not None else np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, dtype=x.dtype, device=x.device)

        # ---- RNN --->
        if True: # Trunc-Wise Implementation, Walk around for L*L complexity.
            (begin, step) = (0, 128)
            mask = np.tril(np.ones(step, step, dtype=x.dtype, device=x.device))[:,:,None,None]   #[l,h,d]
            while begin < l:
                end = begin + step if l-begin>step else l
                truncM = mask[:end-begin,:end-begin]
                truncA, truncX = [item[:, begin:end] for item in [rg, x]]
                cumA = truncA.unsqueeze(2) * truncM                   #[b,l,1,h,d]
                cumA = np.exp(np.cumsum(cumA, dim=1)) * truncM        #[b,l,m,h,d]
                if replicate_num_groups > 1:
                    cumA = np.repeat(cumA, replicate_num_groups, dim=-1)
                shiftB = np.cat([gru_state, truncX[:,:end-begin-1]], dim=1)
                x[:,begin:end] = np.einsum('blmhd,bmhd->blhd', cumA, shiftB) + truncX
                gru_state = x[:,end-1:end]
                begin = end
        elif False: # Approximate version and faster. May cause vanishing gradient
            cumA = np.exp(np.cumsum(rg, dim=1))
            shiftA = np.pad(cumA, (0, 0, 0, 0, 1, -1), value=1.0)
            shiftB = np.cat([gru_state, x[:,:l-1]], dim=1) / (1e-10+shiftA)
            x = np.einsum('blhd,lm,bmhd->blhd', cumA, mask, shiftB) + x
        # <--- RNN ----

        if state is not None:
            state['gru_state'] = x[:,-1:].detach()

        # -- Post Conv -- 
        x = x if not self.post_conv else conv(x, self.conv1d, self.conv_kernel_size, state, 'post_conv_state')

        # Gate and Output
        x = x if self.dropout_v is None else self.dropout_v(x)
        x = x if self.vg_dim == 0 else np.gelu(vg) * x
        x = np.reshape(x, (b, l, -1, self.num_heads))
        x = np.einsum('b l v h , h d v -> b l h d', x, self.out_proj)
        x = x if self.og_dim == 0 else np.gelu(og) * x
        return np.rearrange('b l d v -> b l (d v)', x)
    return __init__(nn.Module(forward = forward),**kwargs)

def DragonflyArgs(name):
    layer = dict(
        name = 'Dragonfly',
        num_heads = 8,
        conv_kernel_size = 4
    )
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [layer]*16,
        resident_gate = True,
        dropout = 0.1,
        bias = False # bias in Linear?
    )
    match(name):
        case 'tiny':
            return args
        case 'vsHawk':
            return dict(
                args,
                latent_dim = 768,
                layers = [dict(
                    name = 'Dragonfly',
                    num_heads = 8,
                )]*16
            )
        case 'HawkOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'Hawk',
                    num_heads = 8
                )]*16,
            )
        case '8m':
            return dict(
                args,
                vocab_dim = 64,
                latent_dim = 768,
                layers = [dict(
                    name = 'Dragonfly',
                    num_heads = 16,
                )]*48
            )
        case '20m':
            return dict(
                args,
                latent_dim = 2048,
                layers = [dict(
                    name = 'Dragonfly',
                    num_heads = 32,
                )]*48
            )
        case '70m':
            return dict(
                args,
                latent_dim = 4096,
                layers = [dict(
                    name = 'Dragonfly',
                    hidden_dim = 8192,
                    num_heads = 64,
                )]*48
            )
        case '200m':
            return dict(
                args,
                latent_dim = 8192,
                layers = [dict(
                    name = 'Dragonfly',
                    hidden_dim = 8192*2,
                    num_heads = 128,
                )]*64
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'Dragonfly-tiny',
        'Dragonfly-vsHawk',
        'Dragonfly-HawkOnly',
        # 'Dragonfly-8m',
        # 'Dragonfly-20m',
        # 'Dragonfly-70m',
        # 'Dragonfly-200m',
    ]
    TrainRoles(roles, lr=6e-3, epochs=2)
    # RunRoles(roles, 'My lord Sebastian')
