import aka.nn as nn
import aka.numpy as np

def Enc(input_dim, dim, n_hiddens, n_classes):
    return nn.Sequential(
            nn.Linear(input_dim + n_classes, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Parrallel(
                nn.Linear(n_hiddens, dim), # 均值
                nn.Linear(n_hiddens, dim), # 方差
            )
        )

def Dec(input_dim, dim, n_hiddens, n_classes):
    return nn.Sequential(
            nn.Linear(dim + n_classes, n_hiddens),
            nn.ReLU(inplace=True),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(inplace=True),
            nn.Linear(n_hiddens, input_dim),
            nn.Sigmoid()
        )
    
def CVAE(input_dim, dim, n_hiddens, n_classes):
    def forward(inputs, targets, *, encoder, decoder):
        # -- flatten --
        inputs = np.flatten(inputs, start_dim=1)

        # -- to one hot --
        targets = np.eye(n_classes)[targets]

        # -- encode --
        (mu, sigma) = encoder(np.cat([inputs, targets], dim=1)) # 将输出分割为均值和方差

        # -- noise --
        # std = np.exp(0.5 * sigma)  # 计算标准差
        # eps = np.randn(std.shape)  # 从标准正态分布中采样噪声
        # z = mu + eps * std  # 重参数化技巧
        eps = np.randn(mu.shape)  # 从标准正态分布中采样噪声
        z = mu + eps * sigma  # 重参数化技巧

        # -- decode --
        outputs = decoder(np.cat([z, targets], dim=1))

        # -- loss --
        kl_divergence = -0.5 * np.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = np.mse_loss(outputs, inputs) + kl_divergence

        return outputs, loss

    return nn.Module(
        forward, 
        encoder = Enc(input_dim, dim, n_hiddens, n_classes),
        decoder = Dec(input_dim, dim, n_hiddens, n_classes)
    )
    
# --- Example ---
if __name__ == "__main__":
    import aka.data as datasets
    nn.train(
        CVAE(input_dim=784, dim=64, n_hiddens=256, n_classes=10),
        datasets.MNIST(),
        mode='withloss',
        batch_size=100,
        epochs=10)
