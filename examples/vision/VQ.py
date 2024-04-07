import aka.nn as nn
import aka.numpy as np

def VectorQuantizer(dim, n_embedding, beta=1.0):
    """
    Inputs the output of the encoder network z and maps it to a discrete
    one-hot vector that is the index of the closest embedding vector e_j
    z (continuous) -> z_q (discrete)
    z.shape = (... , dim)
    referrence: MAGE
    """
    def forward(inputs, targets=None, *, embedding):
        # inputs: [*, C]
        z = inputs
        z_flattened = z.view(-1, dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = np.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            np.sum(embedding.weight**2, dim=1) - 2 * \
            np.matmul(z_flattened, embedding.weight.t())

        indices = np.argmin(d, dim=1)
        z_q = embedding(indices).view(z.shape)

        # compute loss for embedding. TODO. Is nessesary to compute two losses?
        loss = np.mean((z_q.detach() - z)**2) * beta + np.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        return (z_q, indices), loss

    embedding = nn.Embedding(n_embedding, dim)
    embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
    return nn.Module(forward = forward, embedding = embedding)
    
def VectorQuantizer2d(num_heads, embedding_dim, num_embeddings=256, beta=0.25):
    '''
    Multi-Head Quantizer.

    Codebook: [num_embeddings, (num_heads, embedding_dim)]
    '''
    def forward(inputs, targets=None, *, embedding):
        # inputs: [*, C(num_heads,embedding_dim)]
        z = inputs

        # [N,C] --> [N, num_heads, embedding_dim]
        z_flattened = np.reshape(z, (-1, num_heads, embedding_dim))

        # [N, num_heads, embedding_dim] -> [num_heads, N, embedding_dim]
        z_flattened = np.permute(z_flattened, (1,0,2))

        # codebook [num_heads, num_embeddings, embedding_dim]
        codebook = embedding.weight.view((num_heads, num_embeddings, embedding_dim))
        codebook_t = np.permute(codebook, (0,2,1))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # d : [num_heads, N, num_embeddings]
        d = np.sum(z_flattened ** 2, dim=2, keepdim=True) + \
            np.sum(codebook**2, dim=2).view((-1,1,num_embeddings)) - 2 * \
            np.matmul(z_flattened, codebook_t)

        # indices : [num_heads, N]
        indices = np.argmin(d, dim=2)

        # [num_heads, N] --> [N, nuom_heads]
        indices = np.permute(indices, (1, 0))

        # Add index
        indices += np.arange(start=0, end=num_embeddings*num_heads, step=num_embeddings, dtype=indices.dtype)

        # [num_heads, num_embeddings, embedding_dim] --> [(num_embeddings * num_heads), embedding_dim]
        codebook = np.reshape(np.permute(codebook, (1, 0, 2)), (-1, embedding_dim))

        # [N, num_heads]
        indices = np.flatten(indices)

        # indices --> z_q
        z_q = np.index_select(codebook, dim=0, index=indices).view(z.shape)

        # compute loss for embedding. TODO. Is nessesary to compute two losses?
        loss = np.mean((z_q.detach() - z)**2) * beta + np.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        return (z_q, indices), loss


    embedding = nn.Embedding(num_embeddings, embedding_dim*num_heads)
    embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    return nn.Module(forward = forward, embedding = embedding)
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from VQVAE import VQVAE
    from AutoEncoder import AutoEncoder
    nn.train(
        VQVAE(
            vq=VectorQuantizer(dim=12, n_embedding=256),
            ae=AutoEncoder(
                encoder=nn.Conv2d(3, 12, 4, 2, 1), 
                decoder=nn.ConvTranspose2d(12, 3, 4, 2, 1)
            )
        ),
        datasets.ImageFolder("./data/ImageNet", 128),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        mode='withloss',
        show_chart=True,
        batch_size=128,
        epochs=2)
