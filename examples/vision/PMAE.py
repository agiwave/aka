# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from matplotlib import pyplot as plt
from Attentions import SelfAttention, FFAttention
from Embeddings import get_2d_sincos_pos_embed, create_2d_absolute_sin_cos_embedding

def masking(x, mask_ratio=0.4):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    len_mask = L - len_keep
    
    noise = np.rand(N, L)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = np.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = np.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep, _ = np.sort(ids_keep, dim=1)
    ids_mask = ids_shuffle[:, len_keep:]
    ids_mask = np.argsort(ids_mask, dim=1)
    ids_mask, _ = np.sort(ids_mask, dim=1)

    x_masked = np.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_mask = np.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))
    return x_masked, x_mask, len_mask, ids_mask 

def indice(inputs, embedding):
    # inputs: [*, C]
    z = inputs
    z_flattened = z.view(-1, embedding.shape[1])

    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = np.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        np.sum(embedding**2, dim=1) - 2 * \
        np.matmul(z_flattened, embedding.t())

    indices = np.argmin(d, dim=1)
    return indices.view(z.shape[0],z.shape[1])

def forward(self, inputs):
    N, L, D = inputs.shape  # batch, length, dim
    pos_embedding = self.pos_embedding
    encoder = self.encoder
    mask_token = self.mask_token
    x = inputs + pos_embedding.weight
    (x, x_mask, len_mask, ids_mask) = masking(x)
    x = encoder(x)
    x = x[:,:len_mask,:]
    loss = np.mean((x_mask - x) ** 2)
    return x, loss 


def Mixer(input_shape, *, mix_type='L', n_heads, n_hiddens, atten_dim=0, **kwargs):
    (L, D) = input_shape
    match mix_type:
        case 'S':
            # -- Self Attention --
            return nn.Sequential(
                nn.LayerNorm(D),
                SelfAttention((L,D), n_heads=n_heads, atten_dim=atten_dim)
            )
        case 'F':
            # -- FF Attention --
            return nn.Sequential(
                nn.LayerNorm(D),
                FFAttention((L,D), n_heads=4)
            )
        case 'L':
            # -- Linear --
            return nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, n_hiddens),
                nn.ReLU(),
                nn.Linear(n_hiddens, D),
            )

def MAE(encoder_layers='', mask_ratio=0.4, **kwargs):
    (C, H, W) = (3, 224, 224)
    (L, D) = (14*14, 3*16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (14, 14)
    len_keep = int(L * (1 - mask_ratio))
    pos_embedding = nn.Embedding(nPatchH,D)
    pos_embedding.weight = nn.Parameter(
                data=get_2d_sincos_pos_embed(D, nPatchH), 
                requires_grad=False)
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(C, H, W)),

        # -- Patch --
        nn.Reshape(C, nPatchH, PatchH, nPatchW, PatchW),
        nn.Permute(2, 4, 1, 3, 5),   # --> [nPatchH, nPatchW, C, PatchH, PatchW]
        nn.Reshape(L, D),

        # -- Forward --
        nn.Module(
            forward = forward, 
            pos_embedding = pos_embedding,
            encoder = nn.Sequential(
                # -- Layers --
                *[nn.Resident(
                    Mixer(mix_type=mix_type, input_shape=(len_keep,D), **kwargs))
                        for mix_type in encoder_layers]),
            mask_token = nn.Parameter(
                shape=(1,1,D), 
                initializer='zeros',
                requires_grad=False)
        )
    )

if __name__ == "__main__":
    import aka.data as datasets
    # self_losses = nn.train(
    #     MAE(
    #         encoder_layers='SLSLSLSLSL', 
    #         decoder_layers='SLSLSL', 
    #         n_heads=1, atten_dim=256, n_hiddens=256
    #     ), 
    #     datasets.ImageFolder("./data/ImageNet", 224),
    #     persist_filename='MAE_S.pth',
    #     batch_size=64,
    #     epochs=2)

    line_losses = nn.train(
        MAE(
            encoder_layers='FL'*9,
            n_heads=1, n_hiddens=256
        ), 
        datasets.ImageFolder("./data/ImageNet", 224),
        persist_filename='PMAE.pth',
        batch_size=64,
        epochs=2)

    # plt.plot(self_losses)
    plt.plot(line_losses)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend(['Self', 'Linear'], loc='upper right')
    plt.show()

