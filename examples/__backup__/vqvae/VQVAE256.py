import aka.nn as nn
import aka.numpy as np
import types
from VQ import VectorQuantizer2d
from Attentions import SelfAttention
from BasicBlocks import UpSample2d, DownSample2d

def Act(): return nn.Parrallel(np.iden, nn.Functional(np.sigmoid), join_module=np.mul)
def Norm(channels): return nn.GroupNorm(num_groups=32, num_channels=channels)
def ResidentBlock(dim):
    def TokenMixer(): return nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
    def ChannelMixer(): return nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
    return nn.Sequential(
        nn.Resident(nn.Sequential(
            Norm(dim),
            Act(),
            ChannelMixer(),
            Norm(dim),
            Act(),
            TokenMixer(),
        )),
        nn.Resident(nn.Sequential(
            Norm(dim),
            Act(),
            ChannelMixer(),
            Norm(dim),
            Act(),
            ChannelMixer(),
        ))
    )

def Encoder():
    return nn.Sequential(
        DownSample2d([3, 256, 256], 4),
        nn.Conv2d(48, 256, 4, 2, 1),
        ResidentBlock(256),
        # ResidentBlock(256),
        Norm(256),
        Act(),
        nn.Conv2d(256, 256, 4, 2, 1),
        ResidentBlock(256),
        ResidentBlock(256),
        Norm(256),
        Act(),
        nn.Conv2d(256, 256, 3, 1, 1),
    )

def Decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(256, 256, 3, 1, 1),
        ResidentBlock(256),
        ResidentBlock(256),
        Norm(256),
        Act(),
        nn.ConvTranspose2d(256, 256, 4, 2, 1),
        ResidentBlock(256),
        # ResidentBlock(256),
        Norm(256),
        Act(),
        nn.ConvTranspose2d(256, 48, 4, 2, 1),
        UpSample2d([48, 64, 64], 4)
    )

def VQVAE(vq, encoder, decoder):
    def encode(self, inputs):
        # Encode
        z = self.encoder(inputs)

        # [N, C, H, W] --> [N, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()

        # VQ
        (z_q, indics), loss_vq = self.vq(z)

        # [N, H, W, C] --> [N, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return (z_q, indics), loss_vq

    def forward(self, inputs, targets=None):
        # VQ
        (z_q, indics), loss_vq = self.encode(inputs)

        # Decode
        outputs = self.decoder(z_q)

        # Loss
        loss_recons = np.mse_loss(outputs, inputs) 

        print('recons:{:.4f}, vq:{:.4f}'.format(loss_recons.item(),loss_vq.item()))
        return outputs, loss_recons + loss_vq

    return nn.Module(
        forward,
        encode = encode,
        vq = vq,
        encoder = encoder,
        decoder = decoder
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    nn.train(
        # VectorQuantizedVAE(input_dim=3, dim=256, K=512),
        VQVAE(
            encoder=Encoder(), 
            vq=VectorQuantizer2d(num_heads=16, embedding_dim=16, num_embeddings=256),
            decoder=Decoder()),
        datasets.ImageFolder("./data/ImageNet", 256),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        persist_filename='VQVAE256.pth',
        show_chart=True,
        batch_size=100,
        epochs=2)
