import aka.nn as nn
import aka.numpy as np

def XprojBlock(**kwargs):
    '''
    FFN/MLP/Attention/.....
    Args:
        latent_dim:       (required)
        hidden_dim:       latent_dim (default)
        num_heads:        1 (default)
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        bias = getattr(args, 'bias', False)
        dropout = getattr(args, 'dropout', 0.2)

        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', 1)
        self.k_dim = getattr(args, 'k_dim', 0)
        match getattr(args, 'gate', None):
            case 'gh':
                (self.hg_dim, self.og_dim) = (self.hidden_dim, 0)
            case 'go':
                (self.hg_dim, self.og_dim) = (0, args.latent_dim)
            case _:
                (self.hg_dim, self.og_dim) = (0, 0)
        self.act = getattr(np, getattr(args, 'activation', 'gelu'))
        assert args.latent_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_heads == 0
        assert self.k_dim % self.num_heads == 0

        # ik, vk, v, hg, og
        self.in_proj = nn.Parameter(shape=(
            self.num_heads,
            (2 * self.k_dim + self.hidden_dim + self.hg_dim + self.og_dim)//self.num_heads,
            args.latent_dim//self.num_heads)
        )

        # mixers
        def createMixer(name, **kwargs):
            import importlib
            module = importlib.import_module(name)
            short_name = name.split('./\\')[-1]
            m = getattr(module, short_name+"Block", None)
            assert m is not None, f"Unknown layer:{name}"
            return m(**kwargs)
        mixers = getattr(args, 'mixers', None)
        if mixers is not None:
            self.mixers = nn.ModuleList([createMixer(**dict(
                kwargs,
                xproj = False,
                latent_dim = self.hidden_dim,
                **mixerArgs
            )) for mixerArgs in mixers])
        else:
            self.mixers = None

        # o
        self.out_proj = nn.Parameter(shape=(
            self.num_heads,
            args.latent_dim // self.num_heads,
            self.hidden_dim // self.num_heads
        ))
        self.dropout = nn.Dropout(dropout)
        return self

    def copy_xproj_weights(self, in_projs, out_proj):
        in_proj = np.cat(in_projs, dim=0)
        in_proj = np.rearrange('(h d) k->h d k', in_proj, h=self.num_heads)

        out_proj = np.rearrange('(h d) k->h d k', out_proj, h=self.num_heads)
        self.in_proj.copy_(in_proj)
        self.out_proj.copy_(out_proj)

    def proj_in(self, x, **kwargs):
        (b, l, d) = x.shape
        # Inproj
        x = x.view(b, l, self.num_heads, -1)
        xprojs = np.einsum('b l h d, h v d->b l h v', x, self.in_proj)
        split_dims = [self.k_dim, self.k_dim, self.hidden_dim, self.hg_dim, self.og_dim]
        (ik, vk, x, hg, og) = xprojs.split([dim//self.num_heads for dim in split_dims], dim=-1)
        return (ik, vk, x, (hg, og, (b,l,d)))

    def proj_out(self, x, g, **kwargs):
        (hg, og, shape) = g
        (b, l, _) = shape
        x = x if self.dropout is None else self.dropout(x)
        x = self.act(x) if self.hg_dim == 0 else self.act(hg) * x
        x = x.view(b, l, -1, self.num_heads)    # mix heads
        x = np.einsum('b l v h , h d v -> b l h d', x, self.out_proj)
        x = x if self.og_dim == 0 else self.act(og) * x
        return np.reshape(x, shape)

    def forward(self, x, state=None, mixer=None, **kwargs):
        (b, l, d) = x.shape
        (ik, vk, x, g) = self.proj_in(x)

        # mixers
        if self.mixers is not None:
            x_shape = x.shape
            x = np.reshape(x, (b,l,-1))
            if mixer is None:
                for mixer in self.mixers:
                    x = mixer(x, ik=ik, vk=vk, state=state)
            else:
                x = mixer(x, ik=ik, iv=iv, state=state)
            x = x.view(x_shape)

        return self.proj_out(x, g)
    return __init__(nn.Module(forward = forward, proj_in=proj_in, proj_out=proj_out, copy_xproj_weights=copy_xproj_weights),**kwargs)

def XprojArgs(name):
    layer = dict(
        name = 'Xproj',
        num_heads = 8,
        k_dim = 8,
        mixers = [
            dict(
                name = 'Conv1d'
            ),
            dict(
                name = 'Hawk',
            )
        ],
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
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        'Xproj-tiny',
    ]
    TrainRoles(roles, lr=6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
