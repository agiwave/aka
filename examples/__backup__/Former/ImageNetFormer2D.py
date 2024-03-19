
import aka.nn as nn
from aka.nn import Sum, Scale, Resident, Reshape, Transpose

def Mixer(in_size, h_size, mix_dim):
    if(mix_dim==2) :
        return nn.Sequential(                
            nn.BatchNorm1d(256),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size))

    else:
        return nn.Sequential(                
            nn.BatchNorm1d(256),
            nn.Transpose(mix_dim,2),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size),
            nn.Transpose(2,mix_dim))

def ImageNetFormer(num_channels=3, num_classes=10, padding=0):
    return nn.Sequential(
        # Patch
        nn.Reshape(-1, 3, 16, 16, 16, 16),
        nn.Permute(0, 2, 4, 1, 3, 5),
        nn.Reshape(-1, 256, 768),
        Sum(Scale([1.0]),Mixer(768,32,2)),
        Sum(Scale([1.0]),Mixer(256,32,1)),
        Sum(Scale([1.0]),Mixer(768,32,2)),
        Sum(Scale([1.0]),Mixer(256,32,1)),
        Sum(Scale([1.0]),Mixer(768,32,2)),
        Sum(Scale([1.0]),Mixer(256,32,1)),
        Sum(Scale([1.0]),Mixer(768,32,2)),
        Sum(Scale([1.0]),Mixer(256,32,1)),
        nn.BatchNorm1d(256),
        nn.Reshape(-1, 16, 16, 768),
        nn.Transpose(1,3),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(768, num_classes, False),
        nn.Softmax(dim=1)
    )

def Net(**args) :
    return nn.TrainModule(ImageNetFormer(**args), nn.CrossEntropyLoss())
