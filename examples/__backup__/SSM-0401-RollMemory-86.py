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
        self.in_proj = nn.Linear(args.latent_dim, self.num_heads * self.num_states + 2 * self.num_heads + self.hidden_dim + self.gh_dim + self.go_dim, bias=args.bias)
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
        self.delta = nn.Parameter(np.arange(1, 1 + self.num_heads, dtype=np.float))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        self.C = nn.Parameter(shape=(self.num_states, self.num_heads, self.num_states))
        self.norm_v = nn.RMSNorm(self.hidden_dim)
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
        (rkv, q, ig, v, gh, go) = self.in_proj(x).split([
            self.num_states * self.num_heads,
            self.num_heads, 
            self.num_heads,
            self.hidden_dim,
            self.gh_dim, self.go_dim], dim=-1)
        
        # -- Prev Conv -- 
        v = v if not self.prev_conv else conv(v, self.conv1d, self.conv_kernel_size, state, 'prev_conv_state')

        # -- Prepare State --
        ssm_state = None if state is None else state.get('ssm_state',None)
        (t,s0) = ssm_state if ssm_state is not None else (
            0,      # t
            None    # np.zeros(b, 1, self.num_heads, self.num_states, d//self.num_heads, device=x.device)
        )

        q = np.rearrange('b l (h k)->b l h k', q, h=self.num_heads)
        v = np.rearrange('b l (h v)->b l h v', v, h=self.num_heads) * np.sigmoid(ig).unsqueeze(-1)
        # if s0 is not None:
        #     v += np.einsum('blhk,bhkv->blhv', np.silu(q), s0)

        idx = np.arange(t, t+l, device=x.device) % self.num_states
        sx = np.arange(self.num_states, device=x.device)
        C = np.index_select(self.C, dim=0, index=idx)                    #C:[l,h,k]
        idx.unsqueeze(-1)
        B = idx.unsqueeze(-1)-sx                                        #B:[l,h,k]
        B = np.where(B==0,1,0).unsqueeze(1).float()
        mask = np.tril(np.ones(l, l, device=x.device))                  #mask:[l,m]

        # ?? Fix decay or selective decay ?? The performane has no diff.
        rkv = np.repeat(np.exp(-np.softplus(self.delta)).unsqueeze(0), l, dim=0).unsqueeze(0)   # [h] * [b l h]
        # rkv = np.exp(-np.softplus(self.delta) * np.sigmoid(rkv))   # [h] * [b l h]
        cumA = np.cumprod(rkv, dim=1)
        mask = np.tril(np.ones(l, l, device=x.device))[:,1:]
        shiftB = B[:l-1]
        shiftV = v[:,:l-1] / (1e-10+cumA[:,:l-1]).unsqueeze(-1)
        y = np.einsum('lhk,lhk,blhv->blhv',B,C,v)
        ssm_state = np.einsum('lhk,blhv->bhkv',B,v)

        shiftB = np.rearrange('(b m) h (k v)->b m h k v', shiftB, b=1, v=1)
        shiftV = np.rearrange('b m h (k v)->b m h k v', shiftV, k=1)
        cumA = np.rearrange('b l (h k v)->b l h k v', cumA, k=1, v=1)
        C = np.rearrange('(b l) h (k v)->b l h k v', C, b=1, v=1)
        y += np.einsum('bmhkv,bmhkv,lm,blhkv,blhkv->blhv', shiftB, shiftV, mask, cumA, C)
        ssm_state += np.einsum('bmhkv,bmhkv,bhkv,bhkv->bhkv', shiftB, shiftV, cumA[:,-1], C[:,-1])
        if s0 is not None:
            y += np.einsum('blhkv,blhkv,blhkv->blhv', cumA, s0.unsqueeze(1), C)
            ssm_state += cumA[:,-1] * s0
        
        x = np.rearrange('b l h d->b l (h d)', y)
        if state is not None:
            state['ssm_state'] = (t+l, ssm_state.detach())

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
                    num_states = 8,
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
