
import os
os.environ["aka_provider_name"] = "aka.providers.torch"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import aka.nn as nn

def MNIST(num_channels=1, num_classes=10):
    return nn.Sequential(
        nn.Input(nn.ImageShape(1,28,28)),
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

if __name__ == "__main__":
    import aka.numpy as np
    import aka.repo as repo
    datasets = repo.AutoDataset('mnist', token='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')
    def collate_fn(items):
        inputs = np.stack([np.array(item['image'], dtype=np.float) / 255 for item in items], dim=0)
        targets = np.array([item['label'] for item in items])
        return inputs, targets 

    losses = nn.train(
        LeNet(num_classes=10), 
        datasets,
        collate_fn=collate_fn,
        loss_metric=nn.CrossEntropyLoss(), 
        batch_size=64, epochs=5)

    # import aka.data as datasets
    # losses = nn.train(
    #             MNIST(), datasets.MNIST(),
    #             loss_metric=nn.CrossEntropyLoss(), 
    #             batch_size=600, epochs=1)



    from matplotlib import pyplot as plt
    plt.plot(losses)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.show()
