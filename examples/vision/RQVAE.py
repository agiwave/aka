import aka.nn as nn
import aka.numpy as np
import types
from VQ import VectorQuantizer

def RQVAE(vq, ae, n_dims=3):
    def forward(self, inputs, targets=None):
        z = self.ae.encode(inputs)
        # [N, C, H, W] --> [N, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()

        # -- RQVAE implementation. Sum all resident z_q. (4 times.)
        z_q = None
        loss_vq = None
        for i in range(4):
            # VQ
            (zq, indics), zloss = self.vq(z)
            if(i==0):
                z_q = zq
                loss_vq = zloss
            else:
                z_q = zq + z_q
                loss_vq = zloss + loss_vq
            z = z-z_q
        # -- RQVAE implementation end --

        # [N, H, W, C] --> [N, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Decode
        outputs = self.ae.decode(z_q)

        # Loss
        loss_recons = np.mse_loss(outputs, inputs) 

        return outputs, loss_recons + loss_vq

    return nn.Module(
        forward = forward,
        vq = vq,
        ae = ae, 
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from AutoEncoder_F4 import AutoEncoder
    from VQ import VectorQuantizer
    dim = 256
    input_dim = 3  
    n_embedding = 512 
    nn.train(
        RQVAE(
            vq = VectorQuantizer(dim, n_embedding),
            ae = AutoEncoder(input_dim, dim)
        ),
        datasets.ImageFolder("./data/ImageNet", 128),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        show_chart=True,
        batch_size=128,
        epochs=2)
