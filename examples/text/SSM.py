import aka.nn as nn
import aka.numpy as np

def SSMBlock(**kwargs):
    '''
    SSM:
        h.shape = [b, num_heads, k_dim//num_heads, hidden_dim//num_heads]
        h(n) = A(n) * h(n-1) + B(n) * x(n)
        y(n) = C(n) * h(n)   + D(n) * x(n)
             = C(n) * h(n) ...
        alpha = exp(-softplus(delta)*sigmoid(rkv))  range[exp(-softplus(delta)), 1.0]
        B(x(n)) = kv
        C = q
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.k_dim = getattr(args, 'k_dim', self.num_heads*8)
        self.gh_dim = self.hidden_dim if getattr(args, 'v_gate', False) else 0
        self.go_dim = args.latent_dim if getattr(args, 'o_gate', True) else 0
        assert self.hidden_dim % self.num_heads == 0
        assert self.k_dim % self.num_heads == 0

        # A, B, C, gv, v, gh, go
        self.A_mode = getattr(args, 'A_mode', 0)
        self.in_proj = nn.Linear(args.latent_dim, self.k_dim + 3 * self.num_heads + self.hidden_dim + self.gh_dim + self.go_dim, bias=args.bias)
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
        self.C = nn.Parameter(shape=(self.k_dim // self.num_heads, self.num_heads, self.k_dim // self.num_heads))
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
        (A, B, C, v, gv, gh, go) = self.in_proj(x).split([
            self.num_heads, self.num_heads,
            self.k_dim,
            self.hidden_dim,
            self.num_heads, self.gh_dim, self.go_dim], dim=-1)
        
        # -- Prev Conv -- 
        v = v if not self.prev_conv else conv(v, self.conv1d, self.conv_kernel_size, state, 'prev_conv_state')

        # -- Prepare State --
        ssm_state = None if state is None else state.get('ssm_state',None)
        (t,s0) = ssm_state if ssm_state is not None else (
            0,      # t
            None    # np.zeros(b, 1, self.num_heads, self.k_dim//self.num_heads, d//self.num_heads, device=x.device)
        )

        C = np.softmax(np.rearrange('b l (h k)->b l h k', C, h=self.num_heads), dim=-1)
        B = C * np.sigmoid(B).unsqueeze(-1)
        match self.A_mode:
            case 0: # from B
                A = (-np.softplus(self.delta).unsqueeze(-1) * B)   # [h] * [b l h k]
            case 1: # Fixed
                A = self.delta.view(1, 1, self.num_heads, 1)
                A = np.repeat((-np.softplus(A)), l, dim=1)         # [h] * [b l h k]
            case 2: # Indenpent
                A = (-(np.softplus(self.delta) * np.sigmoid(A)).unsqueeze(-1)*C)
            case _:
                assert False
        v = np.rearrange('b l (h v)->b l h v', v, h=self.num_heads) * np.sigmoid(gv).unsqueeze(-1)

        k = np.rearrange('b l h (k v)->b l h k v', B, v=1)
        v = np.rearrange('b l h (k v)->b l h k v', v, k=1)
        A = A.unsqueeze(-1)
        s0 = s0 if s0 is not None else np.zeros(b, 1, self.num_heads, self.k_dim//self.num_heads, d//self.num_heads, device=x.device)
        kv = (1-np.exp(A))*k*v

        # -- RNN --
        if False:# Approximate version and faster. May cause vanishing gradient
            mask = np.tril(np.ones(l, l, device=x.device))
            cumA = np.exp(np.cumsum(A, dim=1))
            shiftA = np.pad(cumA, (0, 0, 0, 0, 0, 0, 1, -1), value=1.0)
            shiftB = np.cat([s0, kv[:,:l-1]], dim=1) / (1e-10+shiftA)
            kv = np.einsum('blhkv,lm,bmhkv->blhkv', cumA, mask, shiftB) + kv
        else:# Accurate Vesion, But expensive.
            mask = np.tril(np.ones(l, l, device=x.device))[:,:,None,None,None]   #[l m h k d]
            cumA = A.unsqueeze(2)*mask
            cumA = np.exp(np.cumsum(cumA, dim=1))*mask
            shiftB = np.cat([s0, kv[:,:l-1]], dim=1)
            kv = np.einsum('blmhkv,bmhkv->blhkv', cumA, shiftB) + kv
        # -- RNN --

        q = np.rearrange('b l h (k v)->b l h k v', C, v=1)
        x = np.einsum('blhkv,blhkv->blhv', q, kv)
        x = np.rearrange('b l h d->b l (h d)', x)
        if state is not None:
            state[ssm_state] = (t+l, kv[:,-1:].detach())

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
        case 'Gemma':
            return dict(
                args,
                prev_norm = 'gemma',
                layers = [
                    dict(
                        name = 'Attention',
                        num_heads = 8,
                        rotary_embedding = True
                    ),
                    dict(
                        name = "MLP",
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
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
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'HawkOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'SSM',
                    num_heads = 8,
                    k_dim = 64,
                    hidden_dim = args['latent_dim']
                )]*16,
            )
        case 'SSM':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'SSM',
                        num_heads = 8,
                        num_states = 8,
                    ),
                    dict(
                        name = "MLP",
                        k_size = args['latent_dim']*3,
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
                    k_dim = 64,
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
                        name = 'Hawk',
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
        case 'RWKV':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'RWKVTMixer',
                        num_heads = 8,
                    ),
                    dict(
                        name = 'RWKVCMixer',
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'RetNet':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Retention',
                        num_heads = 8,
                        rotary_embedding = True
                    ),
                    dict(
                        name = "MLP",
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'SSM-SSM',
        'SSM-SSMOnly',
        'SSM-Gemma',
        'SSM-Hawk',
        'SSM-HawkOnly',
        'SSM-Griffin',
        'SSM-Mamba',
        'SSM-RWKV',
        'SSM-RetNet'
    ]
    TrainRoles(roles, lr = 6e-3, epochs=4)
    # RunRoles(roles, 'My lord Sebastian')
