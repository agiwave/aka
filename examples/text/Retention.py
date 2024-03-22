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

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=use_bias)
        self.group_norm = nn.RMSNorm(self.head_dim)
        return self

    def compute_mask(qlen, klen, latent_dim, num_heads, get_decay_scale=True, retention_mask=None, mode='parallel'):
        decay = np.log(
            1 - 2 ** (-5 - np.arange(num_heads, dtype=np.float))
        )
        if mode == "recurrent":
            retention_info = (decay.view(1, -1, 1, 1).exp(), None, None)
        else:
            # mask = np.tril(np.ones(qlen, klen)).to(decay)
            # mask = np.where(mask <= 0.5, float("inf"), mask)
            index = np.arange(klen).to(decay)
            mask = np.tril(np.ones(klen, klen)).to(decay)
            mask = np.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = np.exp(mask * decay[:, None, None])
            mask = np.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]

            # scaling
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            mask = np.nan_to_num(mask, nan=0.0)

            # decay_scale (used for kv cache)
            if get_decay_scale:
                exponent = np.arange(klen, device=decay.device).float()
                decay_scale = decay.exp().view(-1, 1) ** exponent.view(1, -1)  # [h, t]
                if retention_mask is not None:
                    seqlen = retention_mask.sum(dim=-1)  # [b,]
                    bsz = seqlen.size(0)
                    decay_scale = decay_scale.unsqueeze(0).repeat(bsz, 1, 1)  # [b, h, t]
                    for i, pos in enumerate(seqlen):
                        decay_scale[i, :, pos.item() :] = 0
                else:
                    bsz = 1
                decay_scale = decay_scale.sum(-1).view(bsz, -1, 1, 1)  # [b, h, 1, 1]
            else:
                decay_scale = None

            # mask processing for intra decay
            intra_decay = mask[:, :, -1]
            retention_info = (mask, intra_decay, decay_scale)
        return retention_info

    def recurrent_retention(self, q, k, v, decay, kv_cache=None, retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        kv_cache:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1
        retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1).to(decay)
        else:
            retention_mask = np.ones(k.size(0), 1, 1, 1).to(decay)
        # (b, h, v_dim, qk_dim)
        current_kv = k * v.transpose(-1, -2) * retention_mask

        if kv_cache is not None and "prev_key_value" in kv_cache:
            prev_kv = kv_cache["prev_key_value"]
            prev_scale = kv_cache["scale"]
            scale = np.where(retention_mask == 0, prev_scale, prev_scale * decay + 1)
            # connect prev_kv and current_kv
            # how much to decay prev_kv
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = np.where(retention_mask == 0, 1, decay_amount)
            prev_kv = prev_kv * decay_amount  # decay prev_kv
            current_kv = current_kv / scale.sqrt()  # scale current_kv
            current_kv = np.nan_to_num(
                current_kv, nan=0.0
            )  # remove nan, scale might be 0

            current_kv = prev_kv + current_kv
        else:
            scale = np.ones_like(decay)
            # when retention_mask is 0 at the beginning, setting scale to 1 will
            # make the first retention to use the padding incorrectly. Hence,
            # setting it to 0 here. This is a little ugly, so we might want to
            # change this later. TODO: improve
            scale = np.where(retention_mask == 0, np.zeros_like(decay), scale)

        output = np.sum(q * current_kv, dim=3).unsqueeze(1)  # (b, 1, h, d_v)
        return output, {"prev_key_value": current_kv, "scale": scale}

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

    def forward(self, hidden_states, gctx = {}, state = None, **kwargs): 
        B, T, H = hidden_states.size()
        q, k, v, g = [proj(hidden_states) for proj in [self.q_proj, self.k_proj, self.v_proj, self.g_proj]]
        q, k, v = [t.view(B, T, self.num_heads, -1) for t in [q,k,v]]
        k *= self.scaling

        # -- append kv cache --
        # if state is not None:
        #     window_size = 128
        #     if 'kv_cache' in state:
        #         k_cache, v_cache = state['kv_cache']
        #         if k_cache.size(1) >= window_size:  # Never happen here.
        #             k = np.cat((k_cache[:,1-window_size:,:], k), dim=1)
        #             v = np.cat((v_cache[:,1-window_size:,:], v), dim=1)
        #         else:
        #             k = np.cat((k_cache, k), dim=1)
        #             v = np.cat((v_cache, v), dim=1)
        #         T = k.size(1)
        #     state['kv_cache'] = (k[:,1-window_size:].detach(), v[:,1-window_size:].detach())

        # -- rotary embedding --
        q, k, v = [np.einsum('blnd->bnld', t) for t in [q, k, v]]
        q = apply_rotary_emb(q, gctx)
        k = apply_rotary_emb(k, gctx)

        mode = 'parallel' if T > 1 else 'recurrent'
        if state is not None:
            mask = state.get(mode, None)
            if mask is None or mask[0].size(2) != q.size(2) or mask[0].size(3) != k.size(2):
                mask = compute_mask(q.size(2), k.size(2), self.embed_dim, self.num_heads, mode=mode)
                state[mode] = mask
        else:
            mask = compute_mask(q.size(2), k.size(2), self.embed_dim, self.num_heads, mode=mode)
        (decay_mask, intra_decay, scale) = mask

        # -- Cache load --
        kv_cache = None if state is None else state.get('kv_cache', None)

        if T>1:
            # retention(q,k,v)
            retention = np.einsum('bnld,bnmd->bnlm', q, k)
            retention = retention * decay_mask 
            retention = retention / retention.detach().abs().sum(
                dim=-1, keepdim=True
            ).clamp(min=1, max=5e4)
            retention_out = np.einsum('bnlm,bnmd->blnd', retention, v)

            # kv cache: [b, h, t, v_dim, qk_dim]
            current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
            intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
            current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]
            kv_cache = {"prev_key_value": current_kv, "scale": scale}
        else:
            retention_out, kv_cache = self.recurrent_retention(q,k,v,
                decay_mask,
                kv_cache=kv_cache,
                retention_mask=None #??????
            )

        # -- Cache save --
        if state is not None:
            state['kv_cache'] = kv_cache

        # norm
        normed = self.group_norm(retention_out).reshape(B, hidden_states.size(1), self.value_dim)
        # out gate & proj
        out = self.gate_fn(g) * normed
        return self.out_proj(out)

    return __init__(nn.Module(forward=forward,recurrent_retention=recurrent_retention), args)

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
