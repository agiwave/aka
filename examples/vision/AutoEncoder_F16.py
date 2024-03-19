import aka.nn as nn
import aka.numpy as np

def AutoEncoder(args):
    '''
    Paper VQGAN. Patch_Size(16,16)
    '''
    from AutoEncoder import AutoEncoder as CommonAutoEncoder
    from VQGAN_torch.encoder import Encoder
    from VQGAN_torch.decoder import Decoder
    encoder = nn.Sequential(
        Encoder(args),
        nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
    )
    decoder = nn.Sequential(
        nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device),
        Decoder(args)
    )
    return CommonAutoEncoder(encoder=encoder, decoder=decoder)

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
