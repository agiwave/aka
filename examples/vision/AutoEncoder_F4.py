import aka.nn as nn
import aka.numpy as np

def AutoEncoder(input_dim, dim):
    '''
    Paper VQVAE. Patch_Size(4,4)
    '''
    def Encoder(input_dim, dim):
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

    def Decoder(input_dim, dim):
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

    from AutoEncoder import AutoEncoder as CommonAutoEncoder
    return CommonAutoEncoder(Encoder(input_dim, dim), Decoder(input_dim, dim))

# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    dim = 256
    input_dim = 3  
    n_embedding = 512 
    nn.train(
        AutoEncoder(input_dim=3, dim=dim),
        datasets.ImageFolder("./data/ImageNet", 128),
        optimizer="Adam",
        optimizer_kwargs={'lr':0.0002},
        show_chart=True,
        batch_size=128,
        epochs=2)
