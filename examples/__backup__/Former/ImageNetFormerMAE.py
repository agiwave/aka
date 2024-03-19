
import torch.nn as nn
import torch
from torch.autograd import Function

#
# Transpose模块
#
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return torch.transpose(x, self.dim1, self.dim2).contiguous()

#
# Patch模块
#
class Patch(nn.Module):
    def __init__(self):
        super(Patch, self).__init__()

    def forward(self, x):
        # in  [B, C, H, W]
        #         [0,  1,   2,     3,   4,     5,   6,   7   ]                                    
        # reshape [B,  C,   n_p,   n_e, n_h,   n_p, n_e, n_w ]      - [B, 3, 8, 8, 4, 8, 8, 4]
        # permute [B,  n_p, n_p,   n_e, n_e,   c,   n_h, n_w ]      - [B, 8, 8, 8, 8, 3, 4, 4]
        # reshape [B, (n_p, n_p), (n_e, n_e), (c,   n_h, n_w)]  - [B, (8,8), (8,8), (3,4,4)]

        # n_h * n patchs, patch_size -> [32/n_h, 32/n_w] = [4, 4]
        b, c, h, w = x.size()
        x = x.view(b, c, 8, 8, 4, 8, 8, 4)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
        return x.view(b, 64, 64, 48)

#
# Reshape模块
#
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

#
# 残差网络
#
class Resident(nn.Module):
    def __init__(self, block, scale=None):
        super(Resident, self).__init__()
        self.block = block
        self.scale = scale

    def forward(self, x):
        y = self.block(x)
        if(self.scale != None):
            return x*self.scale + y
        return x+y

#
# 乘法运算
#
class Scale(nn.Module):
    def __init__(self, *args):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(*args))

    def forward(self, x):
        return x * self.weight

#
# 模型相加
#
class Add(nn.Sequential):
    def __init__(self, *args):
        super(Add, self).__init__(*args)

    def forward(self, x):
        y = None
        for i in self._modules:
            if( y == None ):
                y = self._modules[i](x)
            else:
                y = y+self._modules[i](x)
        return y

class Mixer(nn.Sequential):
    def __init__(self, in_size, h_size, mix_dim):
        if(mix_dim==3):
            super(Mixer, self).__init__(
                nn.BatchNorm2d(64),
                nn.Linear(in_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, in_size)
            )
        else:
            super(Mixer, self).__init__(
                nn.BatchNorm2d(64),
                Transpose(mix_dim,3),
                nn.Linear(in_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, in_size),
                Transpose(3,mix_dim),
            )

class Parallel(nn.Sequential):
    def __init__(self, *args):
        super(Parallel, self).__init__(*args)

    def forward(self, *args):
        r = []
        for i in self._modules:
            r.append(self._modules[i](*args))
        return *r
 

class ImageNetFormer(nn.Sequential):
    def __init__(self, num_classes=10):
        super(ImageNetFormer, self).__init__(
            # Patch: (3, 256, 256) -> ((8,8), (8,8), (3,4,4))
            Patch(),
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
            # nn.BatchNorm2d(64),
            # Reshape(16, 16, 16*48),
            # nn.AvgPool2d(kernel_size=(16,16)),
            # Reshape(-1),
            # nn.Linear(16*48, num_classes, False),
            # nn.Softmax(dim=1)
        )

class TrainModule(nn.Module):
    def __init__(self, num_classes=100):
        super(TrainModule, self).__init__()
        self.imageNetFormer = ImageFormer(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        output = self.imageNetFormer(x)
        return self.criterion(output, target)


def Net(**args) :
    return TrainModule(**args)
