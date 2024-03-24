import aka.nn as nn
import aka.numpy as np

def RomeSetArgs(name):
    mlp_args = nn.Args(
        name = 'MLP',
        qk_dim = 384,
        kv_size = 384 * 3,
        kv_gate = False,
    )
    attn_args = nn.Args(
        name = 'Attention',
        windows_size = 128,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
        num_heads = 8,
        num_kv_groups = 8,
        rotary_embedding = True
    )
    args = nn.Args(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [attn_args, mlp_args]*8,
        post_sum_scale = False,
        resident_scale = False,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'vsbase':
            mlp_args.qk_dim = mlp_args.qk_dim
        case 'vsvocabFull':
            args.vocab_dim = args.latent_dim
        case 'vsvocab16':
            args.vocab_dim = 16
        case 'vsqk_dim':
            mlp_args.qk_dim = 64
        case 'vskv_gate':
            mlp_args.kv_gate = True
        case 'vssum_scale':
            args.post_sum_scale = True
        case 'vsresident_scale':
            args.resident_scale = True
        case 'vsAFT':
            attn_args.name = 'AFT'
        case 'vsRet':
            attn_args.name = 'Retention'
        case 'vsTopk':
            mlp_args.activation = 'topk'
        case 'vsBias':
            args.bias = True
        case 'Ret15m':
            attn_args.name = 'Retention'
            args.layers = [attn_args, mlp_args]*11
        case 'AFT15m':
            attn_args.name = 'AFT'
            args.layers = [attn_args, mlp_args]*12
        case 'Gemma15m':
            args.layers = [attn_args, mlp_args]*12
        case '20m':
            args.layers = [attn_args, mlp_args]*15
        case '70m':
            args.layers = [attn_args, mlp_args]*30
            args.latent_dim = 512
            args.attn_args.num_heads = 8
            args.attn_args.num_kv_groups = 8
            mlp_args.kv_size = 512*3
        case _:
            assert False, f"Unknown Gemma name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena, RunArena
    roles = [
        'RomeSet-vsbase',
        'RomeSet-vsvocabFull',
        'RomeSet-vsqk_dim',
        'RomeSet-vsvocab16',          # 200321 - (-4)
        'RomeSet-vskv_gate',
        'RomeSet-vsAFT',
        'RomeSet-vsRet',
        # 'RomeSet-vsresident_scale',   # add score a little bit
        # 'RomeSet-vssum_scale',        # 200321 - (-1)
        # 'RomeSet-vsTopk',             # 200321 - (-2)
        # 'RomeSet-vsBias',             # 200321 - (-3)

        # 'RomeSet-Gemma15mTopk',
        # 'RomeSet-Gemma15m',
        # 'RomeSet-AFT15m',
        # 'RomeSet-Ret15m',
        # 'RomeSet-Gemma15mNOV',
    ]
    # TrainArena(roles, nn.Args(lr = 6e-3, epochs=3))
    RunArena(roles, 'My lord Sebastian')
