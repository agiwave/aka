
import aka.nn as nn
import aka.numpy as np

def RWKVCMixerBlock(args):
    match getattr(args,'RWKV_Ver', '6.0'):
        case '6.0':
            return RWKV_CMix_x060(args)
        case '5.0':
            return RWKV_CMix_x050(args)
        case _:
            return RWKV_CMix_x060(args)

def RWKV_CMix_x060(args):
    def __init__(self, args):
        ratio_1_to_almost0 = 1.0 - 0.5 # 1 - layer_id/n_layer
        ddd = np.ones(1, 1, args.latent_dim)
        for i in range(args.latent_dim):
            ddd[0, 0, i] = i / args.latent_dim
        self.time_maa_k = nn.Parameter(1.0 - np.pow(ddd, ratio_1_to_almost0))
        self.time_maa_r = nn.Parameter(1.0 - np.pow(ddd, ratio_1_to_almost0))

        kv_size = getattr(args, 'kv_size', args.latent_dim)
        self.key = nn.Linear(args.latent_dim, kv_size, bias=args.bias)
        self.receptance = nn.Linear(args.latent_dim, args.latent_dim, bias=args.bias)
        self.value = nn.Linear(kv_size, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, **kwargs):
        xx = np.pad(x, (0,0,1,-1), value=0.) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = self.key(xk)
        k = np.relu(k) ** 2
        kv = self.value(k)
        return np.sigmoid(self.receptance(xr)) * kv
    return __init__(nn.Module(forward=forward), args)

def RWKV_CMix_x050(args):
    def __init__(self, args):
        ratio_1_to_almost0 = 1.0 - 0.5 # 1 - layer_id/n_layer
        ddd = np.ones(1, 1, args.latent_dim)
        for i in range(args.latent_dim):
            ddd[0, 0, i] = i / args.latent_dim
        self.time_mix_k = nn.Parameter(np.pow(ddd, ratio_1_to_almost0))
        self.time_mix_r = nn.Parameter(np.pow(ddd, ratio_1_to_almost0))
        
        kv_size = getattr(args, 'kv_size', args.latent_dim)
        self.key = nn.Linear(args.latent_dim, kv_size, bias=args.bias)
        self.receptance = nn.Linear(args.latent_dim, args.latent_dim, bias=args.bias)
        self.value = nn.Linear(kv_size, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, **kwargs):
        xx = np.pad(x, (0,0,1,-1), value=0.)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = np.relu(k) ** 2
        kv = self.value(k)
        return np.sigmoid(self.receptance(xr)) * kv
    return __init__(nn.Module(forward=forward), args)

