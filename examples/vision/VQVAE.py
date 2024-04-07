import aka.nn as nn
import aka.numpy as np

def VQVAE(vq, ae, n_dims=3):
    '''
    Vector Quantized AE
    '''
    def encode(self, inputs):
        # Encode
        z = self.ae.encode(inputs)

        # [B, C, H, W] --> [B, H, W, C]
        if(self.n_dims == 3):
            z = z.permute(0, 2, 3, 1).contiguous()
            # VQ
            (z_q, indics), loss_vq = self.vq(z)
            # [B, H, W, C] --> [B, C, H, W]
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        else:
            # [B, L, C]
            (z_q, indics), loss_vq = self.vq(z)

        return (z_q, indics), loss_vq

    def decode(self, inputs):
        return self.ae.decode(inputs)

    def forward(self, inputs, targets=None):
        # VQ
        (z_q, indics), loss_vq = self.encode(inputs)

        # Decode
        outputs = self.ae.decoder(z_q)

        # Loss
        loss_recons = np.mse_loss(outputs, inputs) 

        # print('recons:{:.4f}, vq:{:.4f}'.format(loss_recons.item(),loss_vq.item()))
        return outputs, loss_recons + loss_vq

    return nn.Module(
        forward = forward,
        vq = vq,
        ae = ae,
        n_dims = 3,
        encode = encode,
        decode = decode
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from AutoEncoder_F4 import AutoEncoder
    from VQ import VectorQuantizer
    dim = 256
    input_dim = 3  
    embedding_dim = 512 
    nn.train(
        VQVAE(
            vq = VectorQuantizer(dim, embedding_dim),
            ae = AutoEncoder(input_dim, dim)
        ),
        datasets.ImageFolder("./data/ImageNet", 128),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        show_chart=True,
        batch_size=128,
        epochs=2)
