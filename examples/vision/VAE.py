import aka.nn as nn
import aka.numpy as np

def VAE(input_dim, dim, n_hiddens):
    def forward(inputs, targets=None, *, encoder, decoder):
        # -- flatten --
        inputs = np.flatten(inputs, start_dim=1)

        # -- encode --
        (mu, sigma) = encoder(inputs) # 将输出分割为均值和方差

        # -- noise --
        # std = np.exp(0.5 * sigma)  # 计算标准差
        # eps = np.randn(std.shape)  # 从标准正态分布中采样噪声
        # z = mu + eps * std  # 重参数化技巧
        eps = np.randn(mu.shape)  # 从标准正态分布中采样噪声
        z = mu + eps * sigma  # 重参数化技巧

        # -- decode --
        outputs = decoder(z)

        # -- loss --
        kl_divergence = -0.5 * np.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = np.mse_loss(outputs, inputs) + kl_divergence
        return outputs, loss

    return nn.Module(
        forward, 
        encoder = nn.Sequential(
            nn.Linear(input_dim, n_hiddens),
            nn.ReLU(),
            nn.Parrallel(
                nn.Linear(n_hiddens, dim), # 均值
                nn.Linear(n_hiddens, dim), # 方差
            )
        ),
        decoder = nn.Sequential(
            nn.Linear(dim, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, input_dim),
            nn.Sigmoid()
        )
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    trainer = nn.Trainer(VAE(input_dim=784, dim=64, n_hiddens=256),
        # datasets.ImageFolder("./data/ImageNet", 256),
        datasets.MNIST(),
        mode='withloss',
        batch_size=100,
        epochs=10)

    i = 0
    for loss in trainer:
        i += 1
