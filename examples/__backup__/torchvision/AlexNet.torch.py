from torchvision import models

def Net(num_channels=3, num_classes=10, pretrain_weights=False):
    if(pretrain_weights == False):
        return models.AlexNet(num_classes=num_classes)
    
    return models.alexnet(weights=models.AlexNet_Weights,num_classes=num_classes)