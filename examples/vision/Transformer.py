# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from Attention import SelfAttention

def TransFormer(num_classes=100):
    def FeedForword(n_inputs, n_hiddens):
        return nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_inputs),
        )

    (C, H, W) = (3, 224, 224)
    (N, D) = (14*14, 3*16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (14, 14)
    n_atten_heads = 24
    n_ffn_hiddens = 256
    n_layers = 4
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(C, H, W)),

        # -- Patch --
        nn.Reshape(C, nPatchH, PatchH, nPatchW, PatchW),
        nn.Permute(2, 4, 1, 3, 5),   # --> [nPatchH, nPatchW, C, PatchH, PatchW]
        nn.Reshape(N, D),

        # --Position Embedding-- 
        nn.Add(nn.Parameter(shape=(N,D))),

        # -- Layers --
        *[nn.Sequential(
            nn.Resident(SelfAttention((N,D), n_atten_heads)),
            nn.LayerNorm(D),
            nn.Resident(FeedForword(D,n_ffn_hiddens)),
            nn.LayerNorm(D),
        ) for i in range(n_layers)],

        # -- Classifier --
        nn.Permute(2,1),
        nn.AvgPool1d(N),
        nn.Flatten(),
        nn.Linear(D, num_classes)
    )

# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    nn.train(TransFormer(num_classes=100), datasets.ImageFolder("./datasets/ImageNet", 224), batch_size=50, epochs=5)
