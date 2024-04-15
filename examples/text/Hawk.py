import aka.nn as nn
import aka.numpy as np
try:
    from examples.text.CausalScan import CausalScan
    causalScan = CausalScan.apply
except ImportError:
    causalScan = None
    print('Warn: CausalScan import failured.')

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
        self.num_heads = getattr(args, 'num_heads', 8)
        assert self.hidden_dim % self.num_heads == 0
        # rg, ig, v, g
        if getattr(args, 'xproj', True):
            from Xproj import XprojBlock
            self.xproj = XprojBlock(**dict(kwargs, kv_dims=[self.num_heads, self.num_heads, self.hidden_dim]))
        else:
            self.xproj = None

        self.c = nn.Parameter(np.array(-8.), requires_grad=False)
        self.delta = nn.Parameter(np.array(0.5))
        return self

    def forward(self, x, kv=None, state=None, **kwargs):
        (b, l, d) = x.shape
        if self.xproj is not None:
            ((rg, ig), x, go) = self.xproj.proj_in(x, state=state)
        else:
            (rg, ig) = kv

        # -- RG_LRU or GRU --
        x = x.view(b, l, self.num_heads, -1) # np.rearrange('b l (h d)->b l h d', x, h=self.num_heads) # [B,L,H,D]
        rg = ((self.c * np.softplus(self.delta)) * np.sigmoid(rg).unsqueeze(-1))  # [B,L,H,1]
        (x, ig) = (1-np.exp(rg)) * np.sigmoid(ig).unsqueeze(-1) * x, None # The orginal paper: np.sqrt(1-rg**2)*np.sigmoid(ig).unsqueeze(-1) * x
        gru_state = None if state is None else state.get('gru_state',None)

        # ---- RNN --->
        if causalScan is not None:
            gru_state = gru_state if gru_state is not None else np.zeros(b, 1, self.hidden_dim, dtype=x.dtype, device=x.device)
            x = causalScan(gru_state, np.exp(rg).squeeze(-1), x.view(b, l, -1))
            gru_state = x[:,-1:]
        else:
            gru_state = gru_state if gru_state is not None else np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, dtype=x.dtype, device=x.device)

            # Trunc-Wise Implementation, Walk around for L*L complexity.
            (begin, step) = (0, 128)
            mask = np.tril(np.ones(step, step, dtype=x.dtype, device=x.device))[:,:,None,None]   #[l,h,d]
            while begin < l:
                end = begin + step if l-begin>step else l
                maskA = mask[:end-begin,:end-begin]
                truncA, truncX = [item[:, begin:end] for item in [rg, x]]
                cumA = truncA.unsqueeze(2) * maskA                   #[b,l,1,h,d]
                cumA = np.exp(np.cumsum(cumA, dim=1)) * maskA        #[b,l,m,h,d]
                shiftB = np.cat([gru_state, truncX[:,:end-begin-1]], dim=1)
                (x[:,begin:end], cumA) = (np.einsum('blmhd,bmhd->blhd', cumA, shiftB) + truncX, None)
                gru_state = x[:,end-1:end]
                begin = end
            x = np.rearrange('b l h d->b l (h d)',x)
        # <--- RNN ----

        if state is not None:
            state['gru_state'] = gru_state.detach()

        # Gate and Output
        if self.xproj is not None:
            return self.xproj.proj_out(x, go)
        else:
            return x
    return __init__(nn.Module(forward = forward),**kwargs)

if __name__ == "__main__":
    args = dict(
        vocab_dim = 64,
        latent_dim = 384,
        xproj_heads = 4,
        resident_gate = True,
        gate='go',
        activation='silu',
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    att_args = dict(
        num_heads = 8,
        num_states = 2,
        rot_embedding = True,
        mixers = [
            dict(
                name = 'Conv1d'
            )
        ]
    )
    mlp_args = dict(
        name = "Xproj",
        hidden_dim = args['latent_dim']*3,
    )
    roles = [
        # dict( args, name = 'Hawk',
        #     layers = [
        #         dict(att_args, name='Hawk'), mlp_args] * 12
        # ),
        # dict( args, name = 'HawkOnly',
        #     layers = [
        #         dict(att_args, name='Hawk', num_heads=192)] * 24
        # ),
        # dict( args, name = 'Griffin',
        #     layers = [
        #         dict(att_args, name='Hawk'), mlp_args,
        #         dict(att_args, name='Attention'), mlp_args,] * 6
        # ),
        dict( args, name = 'Mamba',
            layers = [
                dict(att_args, name='Mamba')] * 24
        ),
        dict( args, name = 'RetNet',
            layers = [
                dict(att_args, name='Retention'), mlp_args] * 12
        ),
        dict( args, name = 'RWKV',
            layers = [
                dict(
                    name = 'RWKVTMixer',
                    num_heads = 8,
                ),
                dict(
                    name = 'RWKVCMixer',
                    hidden_dim = args['latent_dim']*3,
                )
            ]*12
        ),
        dict( args, name = 'Attn',
            layers = [
                dict(att_args, name='Attention'), mlp_args] * 12
        )
    ]

    from RomeArena import TrainRoles, RunRoles, PlotRoles
    # PlotRoles(roles, np.load('examples/text/hawk-losses.ckt'))
    l = TrainRoles(roles, lr = 6e-3, epochs=1, batch_size=4, show=True, show_frequency=2)
    # RunRoles(roles, 'My lord Sebastian')
