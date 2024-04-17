import aka.nn as nn
import aka.numpy as np
try:
    from CausalScan5d import CausalScan
    causalScan = CausalScan.apply
except ImportError:
    causalScan = None
    print('Warn: CausalScan5d import failed.')

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
        # rg, ig, v, g
        if getattr(args, 'xproj', True):
            from Xproj import XprojBlock
            self.xproj = XprojBlock(**dict(kwargs, kv_dims=[self.k_dim, self.num_heads, self.num_heads, self.num_heads, self.hidden_dim]))
        else:
            self.xproj = None
        assert self.hidden_dim % self.num_heads == 0
        assert self.k_dim % self.num_heads == 0

        # A, B, C, gv, v, gh, go
        self.A_mode = getattr(args, 'A_mode', 0)
        self.delta = nn.Parameter(np.arange(1, 1 + self.num_heads, dtype=np.float))
        return self

    def forward(self, x, kv=None, state=None,  **kwargs):
        (b, l, d) = x.shape
        if self.xproj is not None:
            ((C, A, B, gv), x, go) = self.xproj.proj_in(x, state=state)
        else:
            (C, A, B, gv) = kv

        # -- Prepare State --
        ssm_state = None if state is None else state.get('ssm_state',None)
        (t, ssm_state) = ssm_state if ssm_state is not None else (
            0,      # t
            np.zeros(b, 1, self.num_heads, self.k_dim//self.num_heads, d//self.num_heads, dtype=x.dtype, device=x.device)
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
        A = A.unsqueeze(-1)
        x = np.rearrange('b l (h v)->b l h v', x, h=self.num_heads) * np.sigmoid(gv).unsqueeze(-1)

        if causalScan is not None:
            A = A.unsqueeze(-1)
            B = B.unsqueeze(-2)
            C = C.unsqueeze(-2)
            x, ssm_state = causalScan(ssm_state, A, B, x, C)
        else:
            # -- RNN --
            y = np.empty(x.shape, device=x.device)
            (begin, step) = (0, 64)
            mask = np.tril(np.ones(step, step, dtype=x.dtype, device=x.device))[:,:,None,None,None]   #[l,h,k,d]
            while begin < l:
                end = begin + step if l-begin>step else l
                trilMask = mask[:end-begin, :end-begin]
                (truncA, truncB, truncC, truncV) = [item[:, begin:end] for item in [A,B,C,x]]
                truncB = (1-np.exp(truncA)) * np.einsum('blhk,blhv->blhkv', truncB, truncV)
                truncA = truncA.unsqueeze(2) * trilMask
                truncA = np.exp(np.cumsum(truncA, dim=1)) * trilMask
                shiftB = np.cat([ssm_state, truncB[:, :end-begin-1]], dim=1)
                truncB = np.einsum('blmhkv,bmhkv->blhkv', truncA, shiftB) + truncB
                y[:, begin:end] = np.einsum('blhk,blhkv->blhv', truncC, truncB)
                ssm_state = truncB[:,-1:]
                begin = end
            x, y = y, None
            # -- RNN --

        x = np.rearrange('b l h d->b l (h d)', x)
        if state is not None:
            state['ssm_state'] = (t+l, ssm_state.detach())

        # Gate and Output
        if self.xproj is not None:
            return self.xproj.proj_out(x, go)
        else:
            return x

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
    ssm_args = dict(
        name = 'SSM',
        num_heads = 8,
        mixers = [
            dict(
                name = 'Conv1d'
            ),
        ]
    )
    mlp_args = dict(
        name = "Xproj",
        hidden_dim = args['latent_dim']*3
    )
    match(name):
        case 'Gemma':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Attention',
                        num_heads = 8,
                        rotary_embedding = True
                    ),
                    mlp_args
                ]*8,
            )
        case 'Hawk':
            return dict(
                args,
                layers = [
                    ssm_args,
                    mlp_args
                ]*8,
            )
        case 'SSM':
            return dict(
                args,
                layers = [
                    ssm_args,
                    mlp_args
                ]*8,
            )
        case 'SSMOnly':
            return dict(
                args,
                layers = [dict(
                    ssm_args,
                    k_dim = 64
                )]*16,
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        'SSM-SSM',
        # 'SSM-SSMOnly',
        # 'SSM-Gemma',
        # 'SSM-Hawk',
        # 'SSM-HawkOnly',
        # 'SSM-Griffin',
        # 'SSM-Mamba',
        # 'SSM-RWKV',
        # 'SSM-RetNet'
    ]
    TrainRoles(roles, lr = 6e-3, epochs=4)
    # RunRoles(roles, 'My lord Sebastian')
