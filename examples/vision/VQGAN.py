import aka.nn as nn
import aka.numpy as np

'''
VQGAN = GAN(VQVAE(vq, ae), Discriminator). Nothing here.
'''

# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from VQGAN_torch.encoder import Encoder
    from VQGAN_torch.decoder import Decoder
    from AutoEncoder import AutoEncoder
    from Discriminator import Discriminator
    from VQVAE import VQVAE as CommonVQVAE
    from GAN import GAN
    from VQ import VectorQuantizer

    class ObjArgs():
        def __init__(self, **kwargs): 
            for key in kwargs:
                setattr(self, key, kwargs[key])
    args = ObjArgs(
        latent_dim=256,
        image_size=256,
        num_codebook_vectors=1024,
        beta=0.25,
        image_channels=3,
        dataset_path='./data/ImageNet',
        device="cpu",
        batch_size=3,
        epochs=100,
        learning_rate=2.25e-05,
        beta1=0.5,
        beta2=0.9,
        disc_start=10000,
        disc_factor=1.,
        rec_loss_factor=1.,
        perceptual_loss_factor=1.
    )
    encoder = nn.Sequential(
        Encoder(args),
        nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
    )
    decoder = nn.Sequential(
        nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device),
        Decoder(args)
    )
    nn.train(
        GAN(
            model = CommonVQVAE(        
                vq = VectorQuantizer(args.latent_dim, args.num_codebook_vectors),
                ae = AutoEncoder(encoder, decoder)
            ),
            discriminator = Discriminator(args.image_channels)
        ),
        datasets.ImageFolder(args.dataset_path, args.image_size),
        optimizer="Adam",
        optimizer_kwargs={'lr':args.learning_rate},
        mode='withloss',
        show_chart=True,
        batch_size=args.batch_size,
        epochs=2)

