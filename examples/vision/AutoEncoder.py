import aka.nn as nn
import aka.numpy as np

def AutoEncoder(encoder, decoder):
    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def forward(self, inputs, targets=None):
        outputs = self.decoder(self.encoder(inputs))
        return outputs, np.mse_loss(inputs, outputs)

    return nn.Module(
        forward = forward,
        encoder = encoder,
        decoder = decoder,
        encode = encode,
        decode = decode
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    from AutoEncoder import AutoEncoder
    nn.train(
        AutoEncoder(
            encoder = nn.Conv2d(3, 12, 4, 2, 1),
            decoder = nn.ConvTranspose2d(12, 3, 4, 2, 1)
        ),
        datasets.ImageFolder("./data/ImageNet", 128),
        show_chart=True,
        batch_size=100,
        epochs=2)
