import aka.nn as nn
import aka.numpy as np

def Topk(n_topk, *, dim=-1):
    def forward(self, x):
        dim = self.dim
        n_topk = self.n_topk
        v, indics = np.topk(x, n_topk, dim=dim)
        v = np.select(v, dim=dim, index=n_topk-1).unsqueeze(dim=dim)
        x = np.where(x<v,float('-inf'), x)
        return np.softmax(x, dim=dim)
    return nn.Module(forward=forward, n_topk=n_topk, dim=dim)

def MLPBlock(args):
    '''
    Reference: Gemma, LLaMA
    Common ver:
        (b,l,latent_dim) --up--> (b,l,kv_size, kv_size) --down--> (b, l, latent_dim)
    Full ver:
        (b,l,latent_dim) --in--> (b,l,qk_dim) --up--> (b,l,kv_size, kv_size) 
        --down--> (b, l, hidden_dim) --out--> (b,l,latent_dim)
    Args:
        args.latent_dim,    Required
        args.bias = False,  Optional(False)
        args.mlp_args.kv_size = 384*4, Optional(latent_dim)
        args.mlp_args.kv_gate = False, Optional(False)
        args.mlp_args.qk_dim = 384,  Optional(latetn_dim)
        args.mlp_args.hidden_dim = 384,Optional(latetn_dim)
        args.mlp_args.num_heads = 6,      # not support.
        args.mlp_args.num_kv_groups = 6,  # not support.
    Examples:
        args.mlp_args,kv_gate == True ==> GateMLP
    '''
    def __init__(self, args):
        
        # -- Global Args --
        latent_dim = args.latent_dim
        bias = getattr(args,'bias', False)
        self.post_sum_scale = (lambda x,d:x*(d**-0.5)) if getattr(args, 'post_sum_scale', False) else (lambda x,d:x) 

        # -- MLP Args
        args = args.mlp_args
        kv_size = getattr(args, 'kv_size', latent_dim)
        kv_gate = getattr(args, 'kv_gate', False)
        qk_dim = getattr(args, 'qk_dim', latent_dim)
        hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        act = getattr(args, 'activation', 'gelu')
        match act:
            case 'topk':
                self.act = Topk(*getattr(args, 'activation_args', [3]))
            case _:
                self.act = getattr(np, act)
        self.latent_dim = latent_dim
        self.qk_dim = qk_dim
        self.kv_size = kv_size
        self.hidden_dim = hidden_dim
        self.in_proj = None if qk_dim == latent_dim else nn.Linear(latent_dim, qk_dim, bias=bias)   # Q
        self.up_proj = nn.Linear(qk_dim, kv_size, bias=bias)                                        # K(reversed)
        self.gate_proj = None if not kv_gate else nn.Linear(qk_dim, kv_size, bias=bias)             # G or mask
        self.down_proj = nn.Linear(kv_size, hidden_dim, bias=bias)                                  # V
        self.out_proj = None if hidden_dim == latent_dim else nn.Linear(hidden_dim, latent_dim, bias=bias)
        return self

    def forward(self, x, **kwargs):
        scale = self.post_sum_scale
        x = x if self.in_proj is None else scale(self.in_proj(x), self.latent_dim)
        att = scale(self.up_proj(x), self.qk_dim)
        if(self.gate_proj is not None):
            gate = scale(self.gate_proj(x), self.latent_dim)
            gate = gate if self.act is None else self.act(gate)    # silu LLaMA ?
            att = gate * att
        elif self.act is not None:
            att = self.act(att)
        down = scale(self.down_proj(att), self.kv_size)
        return down if self.out_proj is None else scale(self.out_proj(down), self.hidden_dim)
    return __init__(nn.Module(forward=forward), args)
