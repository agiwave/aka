import aka.nn as nn
import aka.numpy as np

class Args():
    def __init__(self, **kwargs): 
        for key in kwargs: setattr(self, key, kwargs[key])

def RomeSetArgs(name):
    class Tokenizer:
        def __init__(self, path):
            from sentencepiece import SentencePieceProcessor
            self.tokenizer = SentencePieceProcessor('data/Gemma/tokenizer.model')
            self.bos_token_id = self.tokenizer.bos_id()
            self.eos_token_id = self.tokenizer.eos_id()
        def encode(self, s):
            return self.tokenizer.encode(s)
        def decode(self, s):
            return self.tokenizer.decode(s)

    args = Args(
        tokenizer = Tokenizer('data/Gemma/tokenizer.model'),
        vocab_size = 256000,
        vocab_dim = 32,
        latent_dim = 384,
        layers = ['Attention', 'MLP']*8,
        mlp_args = Args(
            qk_dim = 64,
            kv_size = 384 * 3,
            kv_gate = False,
        ),
        attn_args = Args(
            windows_size = 128,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
            num_heads = 8,
            num_kv_groups = 8,
            rotary_embedding = True
        ),
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'Ret15m':
            args.layers = ['Retention', 'MLP']*11
        case 'AFT15m':
            args.layers = ['AFTFull', 'MLP']*12
        case 'Gemma15m':
            args.layers = ['Attention', 'MLP']*12        
        case 'Gemma15mNOV':
            args.layers = ['Attention', 'MLP']*12
            args.vocab_dim = args.latent_dim
        case 'Gemma15mTopk':
            args.layers = ['Attention', 'MLP']*12     
            args.mlp_args.activation = 'topk'
        case '20m':
            args.layers = ['Attention', 'MLP']*15
        case '70m':
            args.layers = ['Attention', 'MLP']*30
            args.latent_dim = 512
            args.attn_args.num_heads = 8
            args.attn_args.num_kv_groups = 8
            args.mlp_args.kv_size = 512*3
        case _:
            assert False, f"Unknown Gemma name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena
    TrainArena([
        'RomeSet-Gemma15mTopk',
        'RomeSet-Gemma15m',
        # 'RomeSet-AFT15m',
        # 'RomeSet-Ret15m',
        # 'RomeSet-Gemma15mNOV',
    ], Args(lr = 6e-4, epochs=3))
