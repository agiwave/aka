import aka.nn as nn
import aka.numpy as np

def SSMBlock(**kwargs):
    '''
    SSM:
        h.shape = [b, num_heads, num_states, hidden_dim//num_heads]
        h(n) = alpha * h(n-1) + (1-alpha) * B(x(n))
        y(n) = C(h(n))
        alpha = exp(-softplus(delta)*sigmoid(rkv))  range[exp(-softplus(delta)), 1.0]
        B(x(n)) = kv
        C = q
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_states = getattr(args, 'num_states', self.hidden_dim)
        self.gh_dim = self.hidden_dim if getattr(args, 'v_gate', False) else 0
        self.go_dim = args.latent_dim if getattr(args, 'o_gate', True) else 0
        self.num_heads = getattr(args, 'num_heads', 8)
        assert self.hidden_dim % self.num_heads == 0

        # rkv, q, k, v, gh, go
        self.in_proj = nn.Linear(args.latent_dim, 3 * self.num_states * self.num_heads + self.hidden_dim + self.gh_dim + self.go_dim, bias=args.bias)
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
        delta = np.arange(1, 1 + self.num_states, dtype=np.float).unsqueeze(0)
        self.delta = nn.Parameter(np.repeat(delta, self.num_heads, dim=0))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self

    def conv(x, k, kernel_size, state, key):
        (b, l, d) = x.shape
        x = np.rearrange('b l d->b d l', x)
        if state is not None:
            conv_state = state.get(key,None)
            if conv_state is not None:
                x = np.cat((state[key], x), dim=2)
            state[key] = x[:, :, (1 - kernel_size):].detach()
        if x.size(2) < l + kernel_size - 1:
            x = np.pad(x, (l + kernel_size - 1 - x.size(2), 0), mode='replicate')
        x = k(x)
        x = np.silu(x)
        return np.rearrange('b d l->b l d', x)

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        n_head_state = self.num_states * self.num_heads
        (rkv, q, k, v, gh, go) = self.in_proj(x).split([
            n_head_state, n_head_state, n_head_state,
            self.hidden_dim,
            self.gh_dim, self.go_dim], dim=-1)
        
        # -- Prev Conv -- 
        v = v if not self.prev_conv else conv(v, self.conv1d, self.conv_kernel_size, state, 'prev_conv_state')

        # -- RG_LRU or GRU --
        rkv = np.rearrange('b l (h k v)->b l h k v', rkv, h=self.num_heads, v=1)
        rkv = np.exp(-np.softplus(self.delta.unsqueeze(-1)) * np.sigmoid(rkv))   # [h num_states] * [b l h num_states]
        
        k = np.rearrange('b l (h k v)->b l h k v', k, h=self.num_heads, v=1)
        v = np.rearrange('b l (h k v)->b l h k v', v, h=self.num_heads, k=1)
        kv = (1-rkv)*np.silu(k)*v

        ssm_state = None if state is None else state.get('ssm_state',None)
        ssm_state = ssm_state if ssm_state is not None else np.zeros(b, 1, self.num_heads, self.num_states, d//self.num_heads, device=x.device)
        
        # -- RNN --
        cumA = np.cumprod(rkv, dim=1)
        mask = np.tril(np.ones(l, l, device=x.device))
        shiftA = np.pad(cumA, (0, 0, 0, 0, 0, 0, 1, -1), value=1.0)
        shiftB = np.cat([ssm_state, kv[:,:l-1]], dim=1) / (1e-10+shiftA)
        kv = np.einsum('blhkv,lm,bmhkv->blhkv', cumA, mask, shiftB) + kv
        # -- RNN --
        
        q = np.rearrange('b l (h k v)->b l h k v', q, h=self.num_heads, v=1)
        x = np.einsum('blhkv,blhkv->blhv', np.silu(q), kv)
        x = np.rearrange('b l h d->b l (h d)', x)
        if state is not None:
            state['ssm_state'] = kv[:,-1:].detach()

        # -- Post Conv -- 
        x = x if not self.post_conv else conv(x, self.conv1d, self.conv_kernel_size, state, 'post_conv_state')

        # Gate and Output
        x = x if self.gh_dim <= 0 else x * np.gelu(gh)
        return self.out_proj(x) if self.go_dim <=0 else self.out_proj(x) * np.gelu(go)
    return __init__(nn.Module(forward = forward),**kwargs)

def SSMArgs(name):
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [dict(
            name = 'SSM',
            num_heads = 8,
            conv_kernel_size = 4
        )]*16,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match(name):
        case 'SSM':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'SSM',
                        num_heads = 8,
                    ),
                    dict(
                        name = "MLP",
                        kv_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'SSMOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'SSM',
                    num_heads = 8,
                    num_states = 1,
                    hidden_dim = args['latent_dim']
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
                        name = 'SSM',
                        num_heads = 8,
                    ),
                ]*8,
            )
        case 'Mamba':
            return dict(
                args,
                layers = [dict(
                    name = 'Mamba',
                    num_heads = 8,
                    num_states = 8
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
        # 'SSM-SSM',
        # 'SSM-Mamba',
        # 'SSM-Griffin',
        'SSM-SSMOnly',
        # 'SSM-SSMOnly',
    ]
    TrainRoles(roles, lr = 6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
