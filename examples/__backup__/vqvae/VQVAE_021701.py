import aka.nn as nn
import aka.numpy as np

def Enc(input_dim, dim):
    return nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            *[nn.Resident(nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            )) for _ in range(2)]
        )

def Dec(input_dim, dim):
    return nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1),
            *[nn.Resident(nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            )) for _ in range(2)],
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

def VectorQuantizer(dim, n_embedding):
    def forward(inputs, targets=None, *, embedding):
        # Implemention by Pytorch-VQVAE
        # from functions import vq_st
        # z_e_x = encoder(inputs)

        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # z_q_x_, indices = vq_st(z_e_x_, embedding.weight.detach())
        # z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        # x_tilde = decoder(z_q_x)

        # # z_q_x_bar_flatten = np.index_select(embedding.weight,
        # #     dim=0, index=indices)
        # # z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        # # z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        # # z_q_x = z_q_x_bar

        # # Reconstruction loss
        # loss_recons = np.mse_loss(x_tilde, inputs)
        # # Vector quantization objective
        # loss_vq = np.mse_loss(z_q_x, z_e_x.detach())
        # # Commitment objective
        # loss_commit = np.mse_loss(z_e_x, z_q_x.detach())

        # loss = loss_recons + loss_vq + loss_commit # * l_w_commitment
        # print('loss_recons: {:.4f}, loss_vq: {:.4f}'.format(loss_recons.item(), loss_vq.item()))
        # return x_tilde, loss

        # -------------------------------------------------------- #
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        referrence: MAGE
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = inputs.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = np.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            np.sum(embedding.weight**2, dim=1) - 2 * \
            np.matmul(z_flattened, embedding.weight.t())

        min_encoding_indices = np.argmin(d, dim=1)
        z_q = embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss_vq = np.mse_loss(z.detach(), z_q)
        loss_commit = np.mse_loss(z, z_q.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss_vq + loss_commit

    embedding = nn.Embedding(n_embedding, dim)
    embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
    return nn.Module(forward, embedding = embedding)
    
def VQVAE(input_dim, dim, n_embedding, l_w_embedding=1, l_w_commitment=0.25):
    def forward(inputs, targets=None, *, embedding, encoder, decoder):
        
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        referrence: MAGE
        """
        z = encoder(inputs)

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = np.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            np.sum(embedding.weight**2, dim=1) - 2 * \
            np.matmul(z_flattened, embedding.weight.t())

        min_encoding_indices = np.argmin(d, dim=1)
        z_q = embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss_vq = np.mse_loss(z.detach(), z_q)
        loss_commit = np.mse_loss(z, z_q.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # decode
        x_hat = decoder(z_q)

        # recons loss
        loss_recons = np.mse_loss(x_hat, inputs) 
        # print('loss_recons: {:.4f}, loss_vq: {:.4f}'.format(loss_recons.item(), loss_vq.item()))
        return x_hat, loss_recons + loss_vq + loss_commit

    embedding = nn.Embedding(n_embedding, dim)
    embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
    return nn.Module(
        forward, 
        embedding = embedding,
        encoder = Enc(input_dim, dim),
        decoder = Dec(input_dim, dim)
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.nn as nn
    import aka.numpy as np
    import aka.data as datasets
    from VQVAE_reference_modules import VectorQuantizedVAE
    nn.train(
        # VectorQuantizedVAE(input_dim=3, dim=256, K=512),
        VQVAE(input_dim=3, dim=256, n_embedding=512),
        datasets.ImageFolder("./data/ImageNet", 128),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        mode='withloss',
        show_chart=True,
        batch_size=128,
        epochs=2)
