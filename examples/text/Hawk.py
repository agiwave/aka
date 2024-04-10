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
        rg = rg.unsqueeze(-1)
        ig = ig.unsqueeze(-1)
        rg = np.sigmoid(rg)
        rg = ((self.c * np.softplus(self.delta)) * rg)  # [B,L,H]

        x = x.view(b, l, self.num_heads, -1) # np.rearrange('b l (h d)->b l h d', x, h=self.num_heads) # [B,L,H,D]
        (x, ig) = (1-np.exp(rg)) * np.sigmoid(ig) * x, None # The orginal paper: np.sqrt(1-rg**2)*np.sigmoid(ig).unsqueeze(-1) * x
        gru_state = None if state is None else state.get('gru_state',None)
        gru_state = gru_state if gru_state is not None else np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, dtype=x.dtype, device=x.device)

        # ---- RNN --->
        if True: # Trunc-Wise Implementation, Walk around for L*L complexity.
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
        elif False: # Approximate version and faster. May cause gradient vanishing
            cumA = np.exp(np.cumsum(rg, dim=1))
            shiftA = np.pad(cumA, (0, 0, 0, 0, 1, -1), value=1.0)
            shiftB = np.cat([gru_state, x[:,:l-1]], dim=1) / (1e-10+shiftA)
            x = np.einsum('blhd,lm,bmhd->blhd', cumA, mask, shiftB) + x
        # <--- RNN ----

        if state is not None:
            state['gru_state'] = x[:,-1:].detach()
        x = x.view(b,l,-1)

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
        dict( args, name = 'Hawk',
            layers = [
                dict(att_args, name='Hawk'), mlp_args] * 12
        ),
        dict( args, name = 'HawkOnly',
            layers = [
                dict(att_args, name='Hawk')] * 24
        ),
        dict( args, name = 'Griffin',
            layers = [
                dict(att_args, name='Hawk'), mlp_args,
                dict(att_args, name='Attention'), mlp_args,] * 6
        ),
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
    l = TrainRoles(roles, lr = 6e-3, epochs=1, show=True, show_frequency=2)
    # RunRoles(roles, 'My lord Sebastian')
