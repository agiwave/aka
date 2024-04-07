# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from matplotlib import pyplot as plt
from Attentions import SelfAttention, FFAttention
from Embeddings import get_2d_sincos_pos_embed, create_2d_absolute_sin_cos_embedding

def masking(x, mask_ratio=0.6):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = np.rand(N, L)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = np.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = np.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = np.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = np.ones([N, L])
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = np.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore

def unmasking(x, ids_restore, *, mask_token):
    # append mask tokens to sequence
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    # x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x = np.cat([x, mask_tokens], dim=1)  # no cls token
    x = np.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    return x

def loss(imgs, pred, mask):
    """
    pred: [N, L, D]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = imgs
    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def forward(self, inputs):
    pos_embedding = self.pos_embedding
    encoder = self.encoder
    decoder = self.decoder
    mask_token = self.mask_token
    x = inputs + pos_embedding.weight
    (x, mask, ids_restore) = masking(x)
    x = encoder(x)
    x = unmasking(x, ids_restore, mask_token=mask_token)
    x = x + pos_embedding.weight
    pred = decoder(x)
    loss = (pred - inputs) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return pred, loss 

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

def MAE(encoder_layers='', decoder_layers='', mask_ratio=0.6, **kwargs):
    (C, H, W) = (3, 224, 224)
    (L, D) = (14*14, 3*16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (14, 14)
    len_keep = int(L * (1 - mask_ratio))
    pos_embedding = nn.Embedding(nPatchH,D)
    pos_embedding.weight = nn.Parameter(
                data=get_2d_sincos_pos_embed(D, nPatchH), 
                # data = create_2d_absolute_sin_cos_embedding(nPatchH, nPatchW, D),
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
            # pos_embedding = nn.Parameter(shape=(L,D)),
            pos_embedding = pos_embedding,
            encoder = nn.Sequential(
                # -- Layers --
                *[nn.Resident(
                    Mixer(mix_type=mix_type, input_shape=(len_keep,D), **kwargs))
                        for mix_type in encoder_layers]),
            decoder = nn.Sequential(
                # -- Layers --
                *[nn.Resident(
                    Mixer(mix_type=mix_type, input_shape=(L,D), **kwargs))
                        for mix_type in decoder_layers]),
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
            encoder_layers='FLFLFLFLFL',
            decoder_layers='FLFLFL', 
            n_heads=1, n_hiddens=256
        ), 
        datasets.ImageFolder("./data/ImageNet", 224),
        persist_filename='MAE_F.pth',
        batch_size=64,
        epochs=2)

    plt.plot(self_losses)
    plt.plot(line_losses)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend(['Self', 'Linear'], loc='upper right')
    plt.show()

