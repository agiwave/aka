import aka.nn as nn
import aka.numpy as np

def PredictionBlock(args):
    '''
        in ----> feat(B,L,feat_dim) --(conv)
             |                          |
             |                 q(B,L,qk_dim) --(K,V)
             |                                  |
             |                              y(B,L,hidden_dim) --(out_proj)-> out
             |                                  |
             |-- v(B,L,hidden_dim) -------------|
                                                |-- loss
    '''
    def __init__(self,args):
        latent_dim = args.latent_dim
        bias = getattr(args,'bias',False)

        args = args.pred_args
        hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        feat_dim = getattr(args,'feat_dim', hidden_dim)
        qk_dim = getattr(args, 'qk_dim', feat_dim)
        kv_size = getattr(args, 'kv_size', latent_dim)
        kv_gate = getattr(args, 'kv_gate', True)
        kernel_size = getattr(args, 'kernel_size', 3)

        self.qk_dim = qk_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(latent_dim, hidden_dim + feat_dim, bias=bias)
        self.out_proj = None if hidden_dim==latent_dim else nn.Linear(hidden_dim, latent_dim, bias=bias)
        self.K = nn.Linear(qk_dim, kv_size, bias=bias)
        self.G = None if not kv_gate else nn.Linear(qk_dim, kv_size, bias=bias)
        self.V = nn.Linear(kv_size, hidden_dim, bias=bias)
        self.Conv = nn.Conv2d(1, qk_dim, kernel_size=(kernel_size, feat_dim), stride=1, padding=0)
        return self

    def forward(self, inputs, targets=False, state = None,**kwargs):
        (B, L, D) = inputs.shape
        (feat_dim, hidden_dim, kernel_size) = self.feat_dim, self.hidden_dim, self.kernel_size

        # in-proj
        x = self.in_proj(inputs)

        # split x
        (feat_x, x) = x.split([feat_dim, hidden_dim], dim=2)

        # conv x
        feat_x = np.pad(feat_x, (0,0,kernel_size-1,0), value=float(0.))
        feat_x = self.Conv(feat_x.unsqueeze(1)) # B, qk_dim, L, 1
        feat_x = np.einsum('bqlw->blq', feat_x)
        feat_x = np.gelu(feat_x)
        feat_x = self.K(feat_x)
        feat_x = np.softmax(feat_x,dim=-1)
        y = self.V(feat_x)
        if self.out_proj is not None:
            y = self.out_proj(y)
        if targets is not None:
            loss = None if L <= 1 else np.mse_loss(y[:,:L-1], x[:,1:])
            return y, loss
        else:
            return y

    return __init__(nn.Module(forward=forward), args)

def PredictionArgs(name):

    args = nn.Object(
        vocab_size = 256000,
        vocab_dim = 32,
        latent_dim = 384,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    attn_args = nn.Object(
        name = 'Attention',
        windows_size = 128,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
        num_heads = 8,
        num_kv_groups = 8,
        rotary_embedding = True
    )
    pred_args = nn.Object(
        name = 'Prediction',
        qk_dim = 64,
        kv_size = 384 * 3,
        kv_gate = False,
        feat_dim = 64,
        kernel_size = 4
    )
    match name:
        case 'base':
            args.layers = [attn_args, pred_args]*6
        case _:
            assert False, f"Unknown Block name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena, RunArena
    TrainArena([
        'Prediction-base'
    ], nn.Object(lr = 6e-4, epochs=3))

    # RunArena([
    #     'Prediction-base'
    # ], "Paul Daniels (born 4 June 1981 in Burlington)")

