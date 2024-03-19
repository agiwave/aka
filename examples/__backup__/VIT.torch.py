from torchvision import models

def Net(num_channels=3, num_classes=10, pretrain_weights=False, **args):
    return models.vit_b_16()