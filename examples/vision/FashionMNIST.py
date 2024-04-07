import os
os.environ["aka_provider_name"] = "aka.providers.torch"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.nn as nn

def FashionMNIST(num_channels=1, num_classes=10):
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(1,28,28)),

        # -- Conv --
        nn.Conv2d(1, 32, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),
        nn.Dropout(0.3),
        nn.Conv2d(32, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),
        nn.Dropout(0.3),

        # -- Classifier --
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

if __name__ == "__main__":
    import aka.data as datasets
    nn.train(
        FashionMNIST(), datasets.FashionMNIST(), 
        loss_metric=nn.CrossEntropyLoss(), 
        batch_size=64, epochs=5)
