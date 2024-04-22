import aka.nn as nn
import aka.numpy as np

def InfiniteAttentionBlock(**kwargs):
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
    
        # -- Attention Args
        self.k_dim = getattr(args, 'k_dim', args.latent_dim)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', 1)
        self.num_kv_groups = getattr(args, 'num_kv_groups', self.num_heads)
        self.group_k_dim = self.k_dim // self.num_heads * self.num_kv_groups
        self.group_v_dim = self.hidden_dim // self.num_heads * self.num_kv_groups
        assert self.k_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_groups == 0
        assert self.hidden_dim % self.num_heads == 0
        self.xproj = getattr(args, 'xproj', True)
        if getattr(args, 'xproj', True):
            from Xproj import XprojBlock
            self.xproj = XprojBlock(**dict(kwargs, kv_dims=[self.k_dim, self.group_k_dim, self.group_v_dim]))
        else:
            self.xproj = None
        self.window_size = getattr(args, 'window_size', None)
        self.scale_dk = (self.group_k_dim//self.num_kv_groups)**-0.5
        self.rot_embedding = getattr(args, 'rotary_embedding', False)
        self.rope_theta = getattr(args, 'rope_theta', 10000)
        self.beta = nn.Parameter(np.arange(self.num_heads, dtype=np.float).unsqueeze(-1))
        return self

    def apply_rotary_emb(x, cache, pos=0):
        '''
        Reference: LLaMA and Gemma
        Applies the rotary embedding to the query and key.
        '''
        B,L,N,D = x.shape
        slen = pos+L
        freqs_cis = cache.get('freqs_cis', None)
        if freqs_cis is None or len(freqs_cis) < slen:
            """Precomputes the frequency cis."""
            freqs = 1.0 / (10000**(np.arange(0, D, 2, dtype=x.dtype, device=x.device)[:(D // 2)] / D))
            t = np.arange(slen, dtype=x.dtype, device=x.device)
            freqs = np.outer(t, freqs)
            freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
            cache['freqs_cis'] = freqs_cis

        y = np.reshape(x, (B,L,N,2,D//2))
        y = np.einsum('blncd->bnldc',y)
        y = np.view_as_complex(y.contiguous())
        y = np.view_as_real(y*freqs_cis[pos:pos+L]).type_as(x)
        y = np.einsum('bnldc->blncd', y)
        return np.reshape(y, (B,L,N,D))

    def causal_mask(shape, dtype, device, *, window_size = None, from_bottomright: bool = False,):
        mask = np.full(shape, dtype=dtype, device=device, fill_value=1)
        shift = 0 if not from_bottomright else shape[-1] - shape[-2] # q_len - k_len
        mask = np.tril(mask, diagonal=shift)
        if window_size is not None:
            mask = np.triu(mask, diagonal=shift - window_size + 1)
        return np.log(mask)

    def forward(self, x, *, kv=None, cache={}, state=None, **kwargs):
        B, L, _ = x.size()

        # -- qkv --
        if self.xproj is not None:
            ((q, k), v, go) = self.xproj.proj_in(x, state=state)
        else:
            ((q, k), v) = (kv, x)

        num_heads, num_kv_groups = self.num_heads, self.num_kv_groups
        q = q.view(B, L, num_heads, -1)
        k = k.view(B, L, num_kv_groups, -1)
        v = v.view(B, L, num_kv_groups, -1)

        # -- rotary embedding --
        if self.rot_embedding:
            q = apply_rotary_emb(q, cache)
            k = apply_rotary_emb(k, cache)

        # -- repeat kv to match q, MQA and GQA --
        if num_kv_groups != num_heads:
            # [B, L, N, D]
            k = np.repeat(k, num_heads // num_kv_groups, dim=2)
            v = np.repeat(v, num_heads // num_kv_groups, dim=2)

        # -- load state --
        if state is not None:
            (M, z) = state.get('Mz', (
                np.zeros(B, self.num_heads, k.size(3), v.size(3), device=q.device, dtype=q.dtype),
                np.ones(B, self.num_heads, k.size(3), device=q.device, dtype=q.dtype)))
        else:
            (M, z) = (
                np.zeros(B, self.num_heads, k.size(3), v.size(3), device=q.device, dtype=q.dtype),
                np.ones(B, self.num_heads, k.size(3), device=q.device, dtype=q.dtype))

        # -- attn --
        x = np.empty_like(v)
        (begin, step) = (0, L if self.window_size is None else self.window_size)
        mask = causal_mask((step, step), q.dtype, q.device, from_bottomright=True)
        while begin < L:
            end = begin + step if L-begin>step else L

            # Trunc Q，K，V
            trunc_q = q[:,begin:end]
            trunc_k = k[:,begin:end]
            trunc_v = v[:,begin:end]

            # Local Attn
            att = np.einsum('blnd,bmnd->bnlm', trunc_q, trunc_k) * self.scale_dk
            att = att + mask[:end-begin, :end-begin]
            att = np.softmax(att, dim=-1)
            trunc_x = np.einsum('bnlm,bmnd->blnd', att, trunc_v)

            # Query Memroy
            act_Q = np.elu(trunc_q) + 1
            norm_z =  np.einsum('blhd,bhd->blh', act_Q, z)
            A = np.einsum('blhd,bhdv->blhv', act_Q, M) / norm_z.unsqueeze(-1)
            beta = np.sigmoid(self.beta)
            x[:,begin:end] = beta * A + (1-beta) * trunc_x

            # Update M,z
            act_k = np.elu(trunc_k) + 1
            norm_z = np.einsum('blhd,bhd->blh', act_k, z)
            norm_x = np.einsum('blhd,bhdv->blhv', act_k, M) / norm_z.unsqueeze(-1)
            z = z + np.einsum('blhd->bhd', act_k)
            M = M + np.einsum('blhd,blhv->bhdv', act_k, trunc_x - norm_x)

            begin = end

        # -- Save state --
        if state is not None:
            state['Mz'] = (M.detach(),z.detach())
        
        # -- output --
        x = x.reshape(B, L, self.hidden_dim)
        if self.xproj is not None:
            return self.xproj.proj_out(x, go)
        else:
            return x
    return __init__(nn.Module(forward=forward), **kwargs)

# --- Example ---
if __name__ == "__main__":
    roles = [
        dict( 
            name = 'InfiAttn',
            vocab_dim = 64,
            latent_dim = 384,
            resident_gate = True,
            gate='go',
            xproj_heads = 4,
            activation='silu',
            dropout = 0.1,
            bias = False, # bias in Linear?
            layers = [
               dict(
                    name = 'InfiniteAttention',
                    num_heads = 8,
                    window_size = 128,
                    rot_embedding = True,
                    mixers = [
                        dict(
                            name = 'Conv1d'
                        )
                    ]
                ), 
                dict(
                    name = "Xproj",
                    hidden_dim = 384*3,
                )
            ] * 8
        )
    ]

    from RomeArena import TrainRoles, RunRoles, PlotRoles
    # PlotRoles(roles, np.load('examples/text/hawk-losses.ckt'))
    l = TrainRoles(roles, lr = 6e-3, epochs=1, batch_size=4, show=True, show_frequency=2)
    # RunRoles(roles, 'My lord Sebastian')
