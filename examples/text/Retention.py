import aka.nn as nn
import aka.numpy as np
from aka.nn import Args

def RetentionBlock(args):
    def __init__(self,args):
        use_bias=args.bias
        self.config = args.attn_args
        self.embed_dim = args.latent_dim
        self.value_dim = getattr(args.attn_args, 'hidden_dim', args.latent_dim)
        self.num_heads = args.attn_args.num_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5
        self.gate_fn = np.silu
        self.window_size = None

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=use_bias)
        self.group_norm = nn.RMSNorm(self.head_dim)
        return self

    def compute_mask(qlen, klen, latent_dim, num_heads):
        # Multi-Scale for diff heads.
        decay = np.log(
            1 - 2 ** (-5 - np.arange(num_heads, dtype=np.float))
        )
        index = np.arange(klen).to(decay)
        mask = np.tril(np.ones(qlen, klen), diagonal=klen-qlen).to(decay)
        mask = np.masked_fill(
            index[-qlen:, None] - index[None, :], ~mask.bool(), float("inf")
        )
        mask = np.exp(mask * decay[:, None, None])
        mask = np.nan_to_num(mask)
        mask = mask.unsqueeze(0)  # [1, h, t, t]
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()

        # scaling
        mask = np.nan_to_num(mask, nan=0.0)

        # decay_scale (used for kv cache)
        scale = decay.exp().view(-1, 1) ** index.view(1, -1)  # [h, t]
        scale = scale.sum(-1).view(1, -1, 1, 1)  # [b, h, 1, 1]
        return (mask, scale)

    def apply_rotary_emb(x, cache, pos=0):
        _,_,L,D = x.shape
        slen = pos+L
        emb = cache.get('rotary_emb', None)
        if emb is None or len(emb[0]) < slen:
            angle = 1.0 / (10000 ** np.linspace(0, 1, D//2))
            angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
            index = np.arange(slen)
            sin = np.sin(index[:, None] * angle[None, :])
            cos = np.cos(index[:, None] * angle[None, :])
            cache['rotary_emb'] = (sin, cos)
        else:
            (sin,cos) = emb
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        y = np.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos[pos:pos+L]) + (y * sin[pos:pos+L])

    def forward(self, hidden_states, gctx={}, state=None, **kwargs): 
        B, T, H = hidden_states.size()
        q, k, v, g = [proj(hidden_states) for proj in [self.q_proj, self.k_proj, self.v_proj, self.g_proj]]
        q, k, v = [t.view(B, T, self.num_heads, -1) for t in [q,k,v]]
        k *= self.scaling

        # -- append kv cache --
        parallel = True if T > 1 else False
        # parallel = True
        # if state is not None:
        #     window_size = self.window_size if self.window_size is not None else 128
        #     if 'cache_kv2' in state:
        #         cache_k, cache_v = state['cache_kv2']
        #         k = np.cat((cache_k, k), dim=1)
        #         v = np.cat((cache_v, v), dim=1)
        #     state['cache_kv2'] = (k[:,1-window_size:].detach(), v[:,1-window_size:].detach())

        # -- rotary embedding --
        q, k, v = [np.einsum('blnd->bnld', t) for t in [q, k, v]]
        q = apply_rotary_emb(q, gctx, pos=k.size(2)-T)
        k = apply_rotary_emb(k, gctx)

        if parallel:
            mask = gctx.get('decay_mask', None)
            if mask is None or mask[0].size(2) != q.size(2) or mask[0].size(3) != k.size(2):
                mask = compute_mask(q.size(2), k.size(2), self.embed_dim, self.num_heads)
                gctx['decay_mask'] = mask
            (decay_mask, scale) = mask

            # retention(q,k,v)
            retention = np.einsum('bnld,bnmd->bnlm', q, k)
            retention = retention * decay_mask 
            retention = retention / retention.detach().abs().sum(
                dim=-1, keepdim=True
            ).clamp(min=1, max=5e4)
            retention_out = np.einsum('bnlm,bnmd->blnd', retention, v)

            if state is not None:
                # kv cache: [b, h, t, v_dim, qk_dim]
                # What's this? :) This could not be S(n) = gamma * S(n-1) + K(n) @ V(n)
                # This is just a approximation.
                current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
                intra_decay = decay_mask[:, :, -1, :, None, None]  # [b, h, t, 1, 1]
                current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]
                state["prev_key_value"] = current_kv
                state["scale"] = scale
        else:
            decay = np.log(
                1 - 2 ** (-5 - np.arange(self.num_heads, dtype=np.float))
            ).view(1, -1, 1, 1).exp()
            current_kv = k * v.transpose(-1, -2)
            if "prev_kv" in state:
                prev_kv = state["prev_kv"]
                prev_scale = state["prev_scale"]
                current_scale = prev_scale * decay + 1
                decay_amount = prev_scale.sqrt() * decay / current_scale.sqrt()
                prev_kv = prev_kv * decay_amount  # decay prev_kv
                current_kv = current_kv / current_scale.sqrt()  # scale current_kv
                current_kv = prev_kv + current_kv
            else:
                current_scale = np.ones_like(decay)

            state["prev_kv"] = current_kv
            state["prev_scale"] = current_scale
            retention_out = np.sum(q * current_kv, dim=3).unsqueeze(1)  # (b, 1, h, d_v)

        # norm
        normed = self.group_norm(retention_out).reshape(B, hidden_states.size(1), self.value_dim)
        out = self.gate_fn(g) * normed
        return self.out_proj(out)
    return __init__(nn.Module(forward=forward), args)

def RetentionArgs(name):
    args = Args(
        vocab_dim = 32,
        latent_dim = 384,
        layers = ['Attention', 'MLP']*8,
        mlp_args = Args(
            qk_dim = 64,
            kv_size = 384 * 3,
            kv_gate = False,
        ),
        attn_args = Args(
            num_heads = 8,
            num_kv_groups = 8,
            rotary_embedding = True,
            window_size = 256,
        ),
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'Ret':
            args.layers = ['Retention', 'MLP']*3
        case _:
            assert False, f"Unknown Ret name{name}"
    return args

def RetNet(name):
    import aka.repo as repo

    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.fopen(name, 'config.json', ftype='json')
    args = Args(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        embedding_scale = True,
        latent_dim = cfg['decoder_embed_dim'],
        lm_head = True,
        prev_norm = 'rms',
        layers = ['Retention', 'MLP']*cfg['decoder_layers'],
        mlp_args = Args(
            kv_size = cfg['decoder_ffn_embed_dim'],
            kv_gate = cfg['use_glu'],
            activation = cfg['activation_fn']
        ),
        attn_args = Args(
            num_heads = cfg['decoder_retention_heads'],
            hidden_dim = cfg['decoder_value_embed_dim'],
        ),
        bias = False,
        dropout = cfg['dropout']
    )

    # -- Model --
    from CausalLM import CausalLM
    model = CausalLM(args)
    if repo.exist(name, "model.safetensors"):
        with repo.fopen(name, "model.safetensors", ftype='safetensor') as f:
            keys = f.keys()
            with np.no_grad():
                model.lm_head.weight.copy_(f.get_tensor(f'lm_head.weight'))
                model.embedding.weight.copy_(f.get_tensor('model.embed_tokens.weight'))
                model.post_norm.weight.copy_(f.get_tensor(f'model.layer_norm.weight'))
                model.prev_norm.weight.copy_(f.get_tensor(f'model.layernorm_embedding.weight'))
                for i in range(len(model.layers)//2):
                    model.layers[i*2].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.retention_layer_norm.weight'))
                    model.layers[i*2].layer.q_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.q_proj.weight'))
                    model.layers[i*2].layer.g_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.g_proj.weight'))
                    model.layers[i*2].layer.k_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.k_proj.weight'))
                    model.layers[i*2].layer.v_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.v_proj.weight'))
                    model.layers[i*2].layer.out_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.out_proj.weight'))
                    model.layers[i*2+1].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.final_layer_norm.weight'))

                    # Take care here. gate and fc1 are just swaped.
                    model.layers[i*2+1].layer.gate_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.fc1.weight'))
                    model.layers[i*2+1].layer.up_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.gate.weight'))
                    model.layers[i*2+1].layer.down_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.fc2.weight'))
    return model

if __name__ == "__main__":
    retnet = RetNet('data/SDPrompt-RetNet-300M')
    print('Model loaded')
    for w in retnet.generator("1girl"):
        print(w, end='')
# <s> 1girl, absurdres, animal ear fluff, animal ears, bangs, bare shoulders, black hair, blue archive, blunt bangs, blush, closed mouth, collarbone, commentary request, eyes visible through hair, green eyes, hair between eyes, halo, hand on own face, hand up, highres, jacket, kisaki blue archive, long hair, long sleeves, looking at viewer, open clothes, open jacket, shinonome asu, simple background, solo, track jacket, upper body, white background, white jacket</s>
    # from RomeArena import TrainArena
    # TrainArena([
    #     # 'Gemma-20m', 
    #     'Retention-Ret',
    # ], Args(lr = 6e-4, epochs=4))
