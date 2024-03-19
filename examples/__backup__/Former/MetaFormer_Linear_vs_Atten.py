# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from SelfAttention import SelfAttention
from matplotlib import pyplot as plt

def LinearMixer(i_layers, input_shape, n_heads, n_hiddens, v_arr=[], **kwargs):
    (N, D) = input_shape
    H = n_heads
    HD = int(D/H)
    assert(HD*H==D)
    if(i_layers in v_arr):
        return nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Reshape(N, H, HD),
            nn.Permute(2, 3, 1),    # [N, H, HD] --> [H, HD, N]
            nn.MatMul(nn.Parameter(shape=(H, N, N))),
            nn.Permute(3, 1, 2),
            nn.Reshape(N,D)
        )
    else:
        return nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, D),
        )

def AttenMixer(i_layers, input_shape, n_heads, n_hiddens, v_arr=[], **kwargs):
    (N, D) = input_shape
    if(i_layers in v_arr):
        return nn.Sequential(
            nn.LayerNorm(D),
            SelfAttention((N,D), n_heads)
        )
    else:
        return nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, D),
        )

def MetaFormer(n_layers, layer, num_classes=100, **kwargs):
    (C, H, W) = (3, 256, 256)
    (N, D) = (16*16, 16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (16, 16)
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(C, H, W)),

        # -- Patch --
        nn.Reshape(C, nPatchH, PatchH, nPatchW, PatchW),
        nn.Permute(2, 4, 1, 3, 5),      # --> [nPatchH, nPatchW, C, PatchH, PatchW]
        nn.Reshape(N, D*3),       
        nn.Linear(D*3, D),              # --> [14*14, 16*16]

        # -- Position Embedding -- 
        nn.Add(nn.Parameter(shape=(N,D))),

        # -- Layers --
        *[nn.Resident(
            layer(i_layers=i, input_shape=(N,D), **kwargs))
                for i in range(n_layers)],

        # -- Pool classifier --
        nn.LayerNorm(D),
        nn.Permute(2,1),
        nn.AvgPool1d(N),

        # -- Mix token classifier --
        # nn.Permute(2,1),
        # nn.Linear(N,1),

        # -- Mix token per channel classifier
        # nn.Permute(2,1),
        # nn.Reshape(D,1,N),
        # nn.MatMul(nn.Parameter(shape=(D,N,1))),

        nn.Flatten(),
        nn.Linear(D, num_classes)
    )

if __name__ == "__main__":
    import aka.data as datasets

    dataset = datasets.ImageFolder("./data/ImageNet", 256)
    kwargs = {
        "batch_size":100,
        "epochs":3
    }
    L0,_ = nn.train(
        MetaFormer(8, LinearMixer, n_heads=1, n_hiddens=256, v_arr=[0,2,4,6]), 
        dataset, **kwargs)
    L2,_ = nn.train(
        MetaFormer(8, LinearMixer, n_heads=16, n_hiddens=256, v_arr=[0,2,4,6]), 
        dataset, **kwargs)
    S2,_ = nn.train(
        MetaFormer(8, AttenMixer, n_heads=16, n_hiddens=256, v_arr=[0,2,4,6]), 
        dataset, **kwargs)

    # L0,_ = nn.train(
    #     MetaFormer(8, LinearMixer, n_heads=4, n_hiddens=256, v_arr=[0,2,4,6]), 
    #     dataset, **kwargs)
    # L2,_ = nn.train(
    #     MetaFormer(8, LinearMixer, n_heads=16, n_hiddens=256, v_arr=[2,4,6]), 
    #     dataset, **kwargs)
    # S2,_ = nn.train(
    #     MetaFormer(8, LinearMixer, n_heads=16, n_hiddens=256, v_arr=[4,6]), 
    #     dataset, **kwargs)

    plt.plot(L0)
    plt.plot(L2)
    plt.plot(S2)
    plt.legend(['L0', 'L2', 'S2'], loc='upper right')
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.show()
