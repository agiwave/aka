# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn

def LeNet(num_channels=3, num_classes=10, padding=0, **args):
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(3,32,32)),

        # -- Conv --
        nn.Conv2d(in_channels = num_channels, out_channels = 6, 
                    kernel_size = 5, stride = 1, padding = padding),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 6, out_channels = 16, 
                    kernel_size = 5, stride = 1, padding = 0),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 16, out_channels = 120, 
                    kernel_size = 5, stride = 1, padding = 0),
        nn.Tanh(),

        # -- Classifier --
        nn.Flatten(),
        nn.Linear(120, 84),
        nn.Tanh(),
        nn.Linear(84, num_classes)
    )

if __name__ == "__main__":
    import aka.repo as repo
    import aka.numpy as np  

    # -- datasets --
    datasets = repo.AutoDataset('cifar10')

    # -- map --
    def collate_fn(items):
        inputs = np.stack([np.array(img, dtype=np.float) / 255 for (img, _) in items], dim=0)
        targets = np.array([label for (_, label) in items])
        return inputs, targets

    # -- train --
    nn.train(
        LeNet(num_classes=10), 
        datasets,
        collate_fn=collate_fn,
        loss_metric=nn.CrossEntropyLoss(), 
        batch_size=64, 
        epochs=5)
