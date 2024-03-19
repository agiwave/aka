# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
from aka.nn import Add, Scale, Resident, Reshape, Parameter

def Mixer(in_size, h_size, mix_dim):
    if(mix_dim==2) :
        return nn.Sequential(                
            nn.BatchNorm1d(),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size))

    else:
        return nn.Sequential(                
            nn.BatchNorm1d(),
            nn.Permute(2,1),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size),
            nn.Permute(2,1))

def ImageFormer(num_channels=3, num_classes=100, padding=0):
    return nn.Sequential(
        # Patch
        nn.Input(nn.ImageShape(3,256,256)),
        nn.Reshape(3, 16, 16, 16, 16),
        nn.Permute(2, 4, 1, 3, 5),
        nn.Reshape(256, 768),
        Add(Scale([1.0]),Mixer(768,32,2)),
        Add(Scale([1.0]),Mixer(256,32,1)),
        Add(Scale([1.0]),Mixer(768,32,2)),
        Add(Scale([1.0]),Mixer(256,32,1)),
        Add(Scale([1.0]),Mixer(768,32,2)),
        Add(Scale([1.0]),Mixer(256,32,1)),
        Add(Scale([1.0]),Mixer(768,32,2)),
        Add(Scale([1.0]),Mixer(256,32,1)),
        nn.BatchNorm1d(),
        nn.Reshape(16, 16, 768),
        nn.Permute(3, 1, 2),
        nn.AvgPool2d(16),
        nn.Flatten(),
        nn.Linear(768, num_classes, False),
    )

if __name__ == "__main__":
    import aka.data as datasets
    nn.train(ImageFormer(num_classes=100), datasets.ImageFolder("./datasets/ImageNet", 256), batch_size=50, epochs=5)
