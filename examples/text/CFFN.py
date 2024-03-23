import aka.nn as nn
import aka.numpy as np

def CFFNBlock(args):
    '''
    CFFN
    Args:
        args.latent_dim 
        args.attn_args.qk_dim(Optional, default: latent_dim)
    '''
    def __init__(self,args):
        # -- Global Args --
        latent_dim = args.latent_dim
        bias = getattr(args, 'bias', False)
        dropout = getattr(args, 'dropout', 0.2)

        # -- Attention Args
        args = args.mlp_args
        self.latent_dim = latent_dim
        self.qk_dim = getattr(args, 'qk_dim', latent_dim)
        self.kv_size = getattr(args, 'kv_size', latent_dim)
        self.kv_gate =  getattr(args, 'kv_gate', False)
        self.hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        self.window_size = getattr(args, 'window_size', None)
        self.num_heads = getattr(args, 'num_heads', 1)
        self.in_proj = None if self.qk_dim == self.latent_dim else nn.Linear(self.latent_dim, self.qk_dim, bias=bias)
        self.conv_size = getattr(args, 'conv_size', 4)
        h_qk_dim = self.qk_dim // self.num_heads
        if self.conv_size > 1:
            self.conv_proj = nn.Conv2d(1, self.kv_size, kernel_size=(self.conv_size, h_qk_dim), stride=(1,h_qk_dim), padding=0)
        else:
            self.up_proj = nn.Parameter(shape=(self.num_heads, self.kv_size, h_qk_dim))
        self.gate_proj = None if not self.kv_gate else nn.Parameter(shape=(self.num_heads, self.kv_size, h_qk_dim))
        act = getattr(args, 'activation', 'gelu')
        self.act = getattr(np, act)
        self.down_proj = nn.Parameter(shape=(self.num_heads, self.kv_size, self.hidden_dim//self.num_heads))
        self.out_proj = None if self.hidden_dim == self.latent_dim else nn.Linear(self.hidden_dim, self.latent_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        return self

    def forward(self, x, *, ctx={}, state=None, **kwargs):
        B, L, _ = x.size()
        x = x if self.in_proj is None else self.in_proj(x)
        if self.conv_size > 1:
            conv = x if self.conv_size == 1 else np.pad(x, (0,0,self.conv_size-1,0))
            conv = conv.unsqueeze(1)
            att = self.conv_proj(conv)
        else:
            up = np.rearrange('b l (h d) -> b l h d', x, h=self.num_heads)
            att = np.einsum('blhd,hmd->bmlh', up, self.up_proj)
        if(self.gate_proj is not None):
            gate = np.rearrange('b l (h d) -> b l h d', x, h=self.num_heads)
            gate = np.einsum('blhd,hmd->bmlh', gate, self.gate_proj)
            gate = gate if self.act is None else self.act(gate)    # silu LLaMA ?
            att = gate * att
        elif self.act is not None:
            att = self.act(att)
        down = np.einsum('bdlh,hdo->bhlo', att, self.down_proj)
        down = np.rearrange('b h l d-> b l (h d)', down)
        return down if self.out_proj is None else self.out_proj(down)
    return __init__(nn.Module(forward=forward), args)

def CFFNArgs(name):
    args = nn.Args(
        vocab_dim = 32,
        latent_dim = 384,
        layers = ['Attention', 'CFFN']*8,
        mlp_args = nn.Args(
            conv_size = 4,
            kv_size = 384 * 3,
            kv_gate = False,
        ),
        attn_args = nn.Args(
            windows_size = 64,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
            qk_dim = 384,
            num_heads = 8,
            num_kv_groups = 8,
            rotary_embedding = True
        ),
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'base':
            args.mlp_args.conv_size = 1
        case 'conv4':
            args.mlp_args.conv_size = 4
        case 'head4':
            args.mlp_args.num_heads = 4
        case 'convh4':
            args.mlp_args.conv_size = 4
            args.mlp_args.num_heads = 4
        case _:
            assert False, f"Unknown name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena
    TrainArena([
        # 'CFFN-conv4',
        # 'CFFN-head4',
        # 'CFFN-convh4',
        'CFFN-base'
    ], nn.Args(lr = 6e-4, epochs=1))
