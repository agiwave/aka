# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
from matplotlib import pyplot as plt
from Attentions import SelfAttention, LinearAttention
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
    x = inputs + pos_embedding
    (x, mask, ids_restore) = masking(x)
    x = encoder(x)
    x = unmasking(x, ids_restore, mask_token=mask_token)
    x = x + pos_embedding
    pred = decoder(x)
    loss = (pred - inputs) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return pred, loss 

def Mixer(i_layers, input_shape, *, n_heads, n_hiddens, atten_dim=0, atten_layers=[], atten_type='Self', **kwargs):
    (L, D) = input_shape
    if(i_layers in atten_layers):
        if(atten_type=='Self'):
            return nn.Sequential(
                nn.LayerNorm(D),
                SelfAttention((L,D), n_heads=n_heads, atten_dim=atten_dim)
            )
        else:
            return nn.Sequential(
                nn.LayerNorm(D),
                LinearAttention((L,D), n_heads=4)
            )
    else:
        return nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, D),
        )

def MAE(encoder_layers=10, decoder_layers=6, mask_ratio=0.6, **kwargs):
    (C, H, W) = (3, 224, 224)
    (L, D) = (14*14, 3*16*16)
    (PatchH, PatchW) = (16, 16)
    (nPatchH, nPatchW) = (14, 14)
    len_keep = int(L * (1 - mask_ratio))
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(C, H, W)),

        # -- Patch --
        nn.Reshape(C, nPatchH, PatchH, nPatchW, PatchW),
        nn.Permute(2, 4, 1, 3, 5),   # --> [nPatchH, nPatchW, C, PatchH, PatchW]
        nn.Reshape(L, D),

        # -- Forward --
        nn.Module(
            forward, 
            # pos_embedding = nn.Parameter(shape=(L,D)),
            pos_embedding = nn.Parameter(
                data=get_2d_sincos_pos_embed(D, nPatchH), 
                # data = create_2d_absolute_sin_cos_embedding(nPatchH, nPatchW, D),
                requires_grad=False),
            encoder = nn.Sequential(
                # -- Layers --
                *[nn.Resident(
                    Mixer(i_layers=i, input_shape=(len_keep,D), **kwargs))
                        for i in range(encoder_layers)]),
            decoder = nn.Sequential(
                # -- Layers --
                *[nn.Resident(
                    Mixer(i_layers=i, input_shape=(L,D), **kwargs))
                        for i in range(decoder_layers)]),
            mask_token = nn.Parameter(
                shape=(1,1,D), 
                initializer='zeros',
                requires_grad=False)
        )
    )

if __name__ == "__main__":
    import aka.data as datasets
    self_losses = nn.train(
        MAE(
            encoder_layers=10, 
            decoder_layers=6, 
            atten_layers=[0,2,4,6,8],
            n_heads=4, atten_dim=16, n_hiddens=256, atten_type='Self'
        ), 
        datasets.ImageFolder("./data/ImageNet", 224),
        persist_filename='MAE_self.pth',
        batch_size=64,
        epochs=4)

    line_losses = nn.train(
        MAE(
            encoder_layers=10,
            decoder_layers=6, 
            atten_layers=[0,2,4,6,8],
            n_heads=4, n_hiddens=256, atten_type='Linear'
        ), 
        datasets.ImageFolder("./data/ImageNet", 224),
        persist_filename='MAE_linear.pth',
        batch_size=64,
        epochs=4)

    plt.plot(self_losses)
    plt.plot(line_losses)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend(['Self', 'Linear'], loc='upper right')
    plt.show()

