import aka.nn as nn
import aka.numpy as np

def Enc(input_dim, dim):
    return nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),
            *[nn.Resident(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            )) for _ in range(2)]
        )

def Dec(input_dim, dim):
    return nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1),
            *[nn.Resident(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            )) for _ in range(2)],
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )
    
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
        """
        z = encoder(inputs)

        # reshape z -> (B, H, W, C) and flatten to (N, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = np.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            np.sum(embedding.weight ** 2, dim=1) - \
            2 * np.matmul(z_flattened, embedding.weight.t())
            
        # find closest encodings
        min_encoding_indices = np.argmin(d, dim=1).unsqueeze(1)
        z_q = embedding(min_encoding_indices).view(z.shape)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # compute loss for embedding
        loss =  np.mse_loss(z.detach(), z_q) * l_w_embedding + \
                np.mse_loss(z, z_q.detach()) * l_w_commitment
        print(loss)

        # perplexity
        # e_mean = np.mean(min_encodings, dim=0)
        # perplexity = np.exp(-np.sum(e_mean * np.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_hat = decoder(z_q)
        loss = loss + np.mse_loss(x_hat, inputs) 

        return x_hat, loss
            # , (
            # # perplexity, 
            # min_encodings, 
            # min_encoding_indices)

    # vq_embedding = nn.Embedding(n_embedding, dim)
    # vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
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
    import aka.data as datasets
    nn.train(
        VQVAE(input_dim=1, dim=256, n_embedding=128),
        # datasets.ImageFolder("./data/ImageNet", 256),
        datasets.MNIST(),
        optimizer="Adam",
        mode='withloss',
        batch_size=300,
        epochs=2)
