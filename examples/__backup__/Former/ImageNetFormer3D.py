
import aka.nn as nn
from aka.nn import Sum, Scale, Resident, Reshape, Transpose, ImageShape

def Mixer(in_size, h_size, mix_dim):
    if(mix_dim==3):
        return nn.Sequential(                
            nn.BatchNorm2d(64),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size)
        )
    else:
        return nn.Sequential( 
            nn.BatchNorm2d(64),
            Transpose(mix_dim,3),
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, in_size),
            Transpose(3,mix_dim),
        )


class ImageNetFormer(nn.Sequential):
    def __init__(self, num_channels=3, num_classes=10, padding=0):
        super(ImageNetFormer, self).__init__(
            # in  [B, C, H, W]
            #         [0,  1,   2,     3,   4,     5,   6,   7   ]                                    
            # reshape [B,  C,   n_p,   n_e, n_h,   n_p, n_e, n_w ]      - [B, 3, 8, 8, 4, 8, 8, 4]
            # permute [B,  n_p, n_p,   n_e, n_e,   c,   n_h, n_w ]      - [B, 8, 8, 8, 8, 3, 4, 4]
            # reshape [B, (n_p, n_p), (n_e, n_e), (c,   n_h, n_w)]  - [B, (8,8), (8,8), (3,4,4)]
            Input(ImageShape(3,256,256)),
            Reshape(-1, 3, 8, 8, 4, 8, 8, 4),
            Permute(0, 2, 5, 3, 6, 1, 4, 7),
            Reshape(-1, 64, 64, 48),

            #Resident(Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(48,16,3)),
            Add(Scale([1.0]),Mixer(64,16,2)),
            Add(Scale([1.0]),Mixer(48,16,3)),
            Add(Scale([1.0]),Mixer(64,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            Add(Scale([1.0]),Mixer(48,16,3)),
            Add(Scale([1.0]),Mixer(64,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            Add(Scale([1.0]),Mixer(64,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            # nn.MaxPool2d(kernel_size=(64,48)),
            nn.BatchNorm2d(64),
            # Reshape(64, 64*48),    #(64，64， 48) -> (64, (64,48))
            # nn.Linear(64*48, 1),   #(64，(64,48)) -> (64, 1)
            # nn.ReLU(),
            # Reshape(-1),
            # nn.Linear(64, num_classes, False),
            Reshape(16, 16, 16*48),
            nn.AvgPool2d(kernel_size=(16,16)),
            nn.Flatten(),
            nn.Linear(16*48, num_classes, False),
            nn.Softmax(dim=1)
        )

def Net(**args) :
    return nn.TrainModule(ImageNetFormer(**args), nn.CrossEntropyLoss())
