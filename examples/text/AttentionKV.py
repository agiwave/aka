import aka.nn as nn
import aka.numpy as np
from aka.nn import Args

try:
    from xformers.ops.fmha import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
except ImportError:
    memory_efficient_attention = None
    LowerTriangularFromBottomRightMask = None

def AttentionKVBlock(args):
    '''
    Group-Query Attention
    Args:
        args.latent_dim 
        args.attn_args.qk_dim(Optional, default: latent_dim)

    Examples:
        default ==> Attention
        args.attn_args.attn_heads = 8 ==> MHA: Multi-Head Attention
        args.attn_args.kv_groups = 1 ==> MQA: Multi-Query Attention
        args.attn_args.kv_groups = 2 ==> GQA: Group-Query Attention
    '''

    # -- Reference: LLaMA and Gemma，--
    def apply_rotary_emb(x, freqs_cis):
        '''
        Reference: LLaMA and Gemma
        Applies the rotary embedding to the query and key tensors.
        '''
        B,L,N,D = x.shape
        y = np.reshape(x, (B,L,N,2,D//2)).float()
        y = np.einsum('blncd->bnldc',y)
        y = np.view_as_complex(y.contiguous())
        y = np.view_as_real(y*freqs_cis).type_as(x)
        y = np.einsum('bnldc->blncd', y)
        return np.reshape(y, (B,L,N,D))

    # -- Reference: LLaMA and Gemma， Could be learned automaticlly? --
    def precompute_freqs_cis(dim: int,
                            end: int,
                            theta: float = 10000.0):
        """Precomputes the frequency cis."""
        freqs = 1.0 / (theta**(np.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = np.arange(end, device=freqs.device)
        freqs = np.outer(t, freqs).float()
        freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def causal_mask(shape, dtype, *, window_size = None, from_bottomright: bool = False,):
        tensor = np.full(shape,dtype=dtype,fill_value=1)
        num_queries, num_keys = shape[-2:]
        shift = 0
        if from_bottomright:
            shift = num_keys - num_queries

        mask = np.tril(tensor, diagonal=shift)
        if window_size is not None:
            mask = np.triu(mask, diagonal=shift - window_size + 1)
        return np.log(mask)

    def forward(self, x, *, gctx={}, state=None):
        B, L, _ = x.size()

        # -- qkv --
        attn_qk_dim, attn_heads, attn_kv_groups = self.attn_qk_dim, self.attn_heads, self.attn_kv_groups
        attn_head_dim = attn_qk_dim // attn_heads
        attn_k_dim, attn_v_dim = self.attn_k_dim, self.attn_v_dim
        q, k, v  = self.in_proj(x).split([attn_qk_dim, attn_k_dim, attn_v_dim], dim=2)
        q = q.view(B, L, attn_heads, attn_head_dim)
        k = k.view(B, L, attn_kv_groups, -1)
        v = v.view(B, L, attn_kv_groups, -1)

        # -- rotary embedding --
        M = k.size(1)
        if self.rot_embedding:
            if 'freqs_cis' in gctx:
                freqs_cis = gctx['freqs_cis']
                if freqs_cis.size(0) < M:
                    freqs_cis = precompute_freqs_cis(attn_head_dim,M,theta=self.rope_theta)
                    gctx['freqs_cis'] = freqs_cis
            else:
                freqs_cis = precompute_freqs_cis(attn_head_dim,M,theta=self.rope_theta)
                gctx['freqs_cis'] = freqs_cis

            q = apply_rotary_emb(q, freqs_cis=freqs_cis[M-L:M])
            k = apply_rotary_emb(k, freqs_cis=freqs_cis[:M])

        # -- repeat kv to q, MQA and GQA --
        if attn_kv_groups != attn_heads:
            # [B, L, N, head_dim]
            k = np.repeat(k, attn_heads // attn_kv_groups, dim=2)
            v = np.repeat(v, attn_heads // attn_kv_groups, dim=2)

        # -- append state cache --
        if state is not None:
            window_size = self.window_size if self.window_size is not None else 128
            if 'kv_cache' in state:
                k_cache, v_cache = state['kv_cache']
                if k_cache.size(1) >= window_size:  # Never happen here.
                    k = np.cat((k_cache[:,1-window_size:,:], k), dim=1)
                    v = np.cat((v_cache[:,1-window_size:,:], v), dim=1)
                else:
                    k = np.cat((k_cache, k), dim=1)
                    v = np.cat((v_cache, v), dim=1)
            state['kv_cache'] = (k[:,1-window_size:].detach(), v[:,1-window_size:].detach())

        # -- attn --
        if memory_efficient_attention is not None:
            if self.window_size is None:
                y = memory_efficient_attention(q,k,v, attn_bias=LowerTriangularFromBottomRightMask())
            else:
                y = memory_efficient_attention(q,k,v, attn_bias=LowerTriangularFromBottomRightLocalAttentionMask(self.window_size))
        else:
            att = np.einsum('blnd,bmnd->bnlm', q, k) * self.scale_dk
            att = att + causal_mask((L,M), q.dtype, window_size=self.window_size, from_bottomright=True)
            att = np.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = np.einsum('bnlm,bmnd->blnd', att, v)

        if state is not None:
            # window_size = self.window_size if self.window_size is not None else 128
            state['kv_cache'] = (k[:,1-window_size:].detach(), y[:,1-window_size:].detach())

        y = y.reshape(B, L, attn_qk_dim) # re-assemble all head outputs side by side
        return self.resid_dropout(self.out_proj(y))

    # -- Global Args --
    latent_dim = args.latent_dim
    bias = getattr(args, 'bias', False)

    # -- Attention Args
    args = args.attn_args
    attn_qk_dim = getattr(args, 'qk_dim', latent_dim)
    attn_hidden_dim = getattr(args, 'hidden_dim', latent_dim)
    attn_heads = getattr(args, 'num_heads', 1)
    attn_kv_groups = getattr(args, 'num_kv_groups', attn_heads)
    bias = getattr(args, 'bias', False)
    dropout = getattr(args, 'dropout', 0.2)
    attn_head_dim = attn_qk_dim//attn_heads
    k_dim = attn_head_dim * attn_kv_groups
    v_dim = attn_hidden_dim//attn_heads * attn_kv_groups
    window_size = getattr(args, 'window_size', None)
    assert attn_head_dim * attn_heads == attn_qk_dim
    assert attn_heads % attn_kv_groups == 0
    return nn.Module( 
        forward = forward, 
        in_proj = nn.Linear(latent_dim, attn_qk_dim + k_dim + v_dim, bias=bias),
        out_proj = nn.Linear(attn_hidden_dim, latent_dim, bias=bias),
        attn_dropout = nn.Dropout(dropout),
        resid_dropout = nn.Dropout(dropout),
        window_size = window_size,
        attn_qk_dim = attn_qk_dim,
        attn_k_dim = k_dim,
        attn_v_dim = v_dim,
        attn_heads = attn_heads,
        attn_kv_groups = attn_kv_groups,
        scale_dk = 1.0/np.sqrt(np.array([attn_head_dim])),
        rot_embedding = getattr(args, 'rotary_embedding', False),
        rope_theta = getattr(args, 'rope_theta', 10000)
    )

# --- Example ---
if __name__ == "__main__":
    atten = AttentionKVBlock(Args(
        latent_dim = 384,
        attn_args = Args(
            window_size = 128,
            hidden_dim = 256,
            qk_dim = 384,
            num_heads = 6,
            num_kv_groups = 3,
        )
    ))
    input = np.randn(50, 100, 384)
    output = atten(input)
    print(output.size())