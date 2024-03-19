
import aka.nn as nn

def AlexNet(num_channels=3, num_classes=100):
    '''AlexNet

    Remark:
        Input shape of AlexNet should be [B, C, H, W] = [Batch Size, Channels, Height, Weight]

    Args:
        num_channels: Image channels. It should be 1 or 3
        num_classes: The output tensor size. One hot category tensor.
    '''
    stride=4
    padding=2
    return nn.Sequential(
        # -- Input --
        nn.Input(nn.ImageShape(3, 224, 224)),
        
        # -- Conv ---
        nn.Conv2d(num_channels,48,kernel_size=11,stride=stride,padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(48,128, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),

        # -- Classifier --
        nn.Flatten(),
        nn.Linear(6*6*128,2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048,2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048,num_classes)
    )

if __name__ == "__main__":
    import aka.data as datasets
    nn.train(
        AlexNet(), 
        datasets.ImageFolder(root="./data/ImageNet", resize=224), 
        loss_metric=nn.CrossEntropyLoss(), 
        batch_size=64, epochs=5)
