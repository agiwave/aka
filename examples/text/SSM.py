
import aka.nn as nn
import aka.numpy as np
from aka.nn import Args

def SSMBlock(args):
    '''

    '''
    def __init__(self, args):
        latent_dim = args.latent_dim
        bias = getattr(args, 'bias', False)
        args = args.attn_args

        hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        self.hidden_dim = hidden_dim

        num_states = getattr(args, 'num_states', 64)
        self.num_states = num_states

        self.in_proj = nn.Linear(latent_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, latent_dim, bias=bias)

        window_size = getattr(args, 'window_size', 128)
        self.window_size = window_size

        self.A = nn.Parameter(shape=(num_states, num_states), initializer='xavier_uniform')
        self.B = nn.Parameter(shape=(window_size, num_states), initializer='xavier_uniform')
        self.C = nn.Parameter(shape=(hidden_dim, num_states), initializer='xavier_uniform')
        self.D = nn.Parameter(np.ones(hidden_dim))
        self.mask = np.tril(np.ones(window_size, window_size)).unsqueeze(-1)

        dropout = getattr(args, 'dropout', 0.2)
        self.resid_dropout = nn.Dropout(dropout)
        return self

    def ssm_state_v(A,B,h,x):
        return np.einsum('mn,bnd->bmd', A, h) + np.einsum('ln,bld->bnd', B, x)
    def ssm(A,B,C,h,x,mask):
        x = x*mask
        h = ssm_state_v(A,B,h,x*mask)
        return np.einsum('dn,bnd->bd', C, h)
    vmap_ssm = np.vmap(ssm, in_dims=(None,None,None,None,None,0), out_dims=(1))

    def forward(self, inputs, *, gctx={}, state=None, **kwargs):
        outputs = None
        window_size = self.window_size
        (hidden_dim, num_states) = self.hidden_dim, self.num_states
        while inputs is not None:
            B, L, D = inputs.shape
            if L > window_size:
                (x, inputs) = inputs.split([window_size, L-window_size], dim=1)
                L = window_size
            else:
                (x, inputs) = inputs, None
            
            ssmA, ssmB, ssmC, ssmD = self.A, self.B, self.C, self.D
            if state is not None:
                ssm_state = state.get('ssm_state', None)
                if ssm_state is None:
                    ssm_state = np.zeros((B,num_states,D))
                y = vmap_ssm(ssmA, ssmB, ssmC, ssm_state, x, self.mask[:L,:L])
                y = y + ssmD * x
                ssm_state = ssm_state_v(ssmA, ssmB, ssm_state, x)
                state['ssm_state']=ssm_state.detach()
            else:
                ssm_state = np.zeros((B,num_states,D))
                y = vmap_ssm(ssmA, ssmB, ssmC, ssm_state,x,self.mask[:L,:L])
                y = y + ssmD * x
            outputs = y if outputs is None else np.cat([outputs, y], dim=1)
        return self.resid_dropout(self.out_proj(outputs))
    return __init__(nn.Module(forward=forward), args)

def SSMArgs(name):
    args = Args(
        vocab_dim = 32,
        latent_dim = 384,
        layers = ['SSM', 'MLP']*8,
        mlp_args = Args(
            qk_dim = 64,
            kv_size = 384 * 3,
            kv_gate = False,
        ),
        attn_args = Args(
            windows_size = 64,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
            qk_dim = 384,
            num_heads = 8,
            num_kv_groups = 8,
            rotary_embedding = True,
            num_states = 64
        ),
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'Base':
            args.layers = ['SSM', 'MLP']*6
        case _:
            assert False, f"Unknown SSM name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena
    TrainArena([
        # 'Gemma-20m', 
        'SSM-Base',
    ], Args(lr = 6e-4, epochs=4))
