# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from Attention import SelfAttention

def FeedForword(n_inputs, n_hiddens):
    return nn.Sequential(
        nn.Linear(n_inputs, n_hiddens),
        nn.ReLU(),
        nn.Linear(n_hiddens, n_inputs),
    )

def TokenMixer(input_shape, n_heads):
    (N, D) = input_shape
    H = n_heads
    HD = int(D/H)
    assert(HD*H==D)
    return nn.Sequential(
        nn.Linear(D, D),
        nn.ReLU(),
        nn.Reshape(N, H, HD),
        nn.Permute(2, 3, 1),    # [N, H, HD] --> [H, HD, N]
        nn.MatMul(nn.Parameter(shape=(H, N, N))),
        nn.Permute(3, 1, 2),
        nn.Reshape(N,D)
    )


def MetaFormer(num_classes=100):
    (C, H, W) = (3, 224, 224)
    (N, D) = (14*14, 3*16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (14, 14)
    n_atten_heads = 32
    n_ffn_hiddens = 32
    n_layers = 4
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(C, H, W)),

        # -- Patch --
        nn.Reshape(C, nPatchH, PatchH, nPatchW, PatchW),
        nn.Permute(2, 4, 1, 3, 5),   # --> [nPatchH, nPatchW, C, PatchH, PatchW]
        nn.Reshape(N, D),

        # -- Position Embedding -- 
        nn.Add(nn.Parameter(shape=(N,D))),

        # -- Layers --
        *[nn.Sequential(
            # nn.Resident(SelfAttention((N,D), n_atten_heads)),
            nn.Resident(TokenMixer((N,D), n_atten_heads)),
            nn.LayerNorm(D),
            nn.Resident(FeedForword(D,n_ffn_hiddens)),
            nn.LayerNorm(D),
        ) for i in range(n_layers)],

        # -- Pool classifier --
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
    nn.train(MetaFormer(num_classes=100), datasets.ImageFolder("./data/ImageNet", 224), batch_size=50, epochs=5)
