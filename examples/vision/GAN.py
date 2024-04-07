import aka.nn as nn
import aka.numpy as np

def GAN(model, discriminator, disc_factor=1.0):
    '''
    GAN from VQGAN
    '''
    def forward(inputs, targets=None, *, model, discriminator):
        outputs = model(inputs)
        if(isinstance(outputs, tuple)):
            q_loss = outputs[-1]
            outputs = outputs[0]
        else:
            q_loss = 0.

        disc_real = discriminator(inputs)
        disc_fake = discriminator(outputs)

        # ???? -np.mean from VQGAN ???
        # g_loss = -np.mean(disc_fake)
        g_loss = np.mean(disc_fake)
        q_loss = q_loss + disc_factor * g_loss

        d_loss_real = np.mean(np.relu(1. - disc_real))
        d_loss_fake = np.mean(np.relu(1. + disc_fake))
        gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

        return outputs, ((model, q_loss), (discriminator, gan_loss))

    return nn.Module(
        forward = forward, 
        model = model,
        discriminator = discriminator
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from VQGAN_torch.encoder import Encoder
    from VQGAN_torch.decoder import Decoder
    from AutoEncoder import AutoEncoder
    from Discriminator import Discriminator
    from VQVAE import VQVAE as CommonVQVAE
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
