
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
    def __init__(self, n_h, n_w):
        super(Patch, self).__init__()
        self.n_h = n_h
        self.n_w = n_w

    def forward(self, x):
        # in  [B, C, H, W] 
        # reshape --> [B, C, n_h, H/n_h, n_w, W/n_w] --> [-1, 3, 8, 4, 8, 4]
        # permute --> [B, n_h, n_w, C, H/n_h, W/n_w] --> [-1, 8, 8, 3, 4, 4]
        # reshape --> [B, n_h|n_w, C|H/n_h, W/n_w]   --> [-1, 64, 48]

        # n_h * n patchs, patch_size -> [32/n_h, 32/n_w] = [4, 4]
        n_h = self.n_h
        n_w = self.n_w
        b, c, h, w = x.size()
        patch_h = (int)(h/n_h)
        patch_w = (int)(w/n_w)
        x = x.view(b, c, n_h, patch_h, n_w, patch_w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(b, n_h*n_w, c*patch_h*patch_w)

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
# 矩阵相乘
#
class MatMul(nn.Module):
    def __init__(self, *args):
        super(MatMul, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(*args))
        nn.init.kaiming_normal_(self.weight.data)

    def forward(self, x):
        return torch.matmul(x, self.weight)

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
        if(mix_dim==2):
            super(Mixer, self).__init__(
                nn.BatchNorm1d(64),
                MatMul(in_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, in_size)
            )
        else:
            super(Mixer, self).__init__(
                nn.BatchNorm1d(64),
                Transpose(1,2),
                nn.Linear(in_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, in_size),
                Transpose(2,1),
            )


class MixFormer(nn.Sequential):
    def __init__(self, num_channels=3, num_classes=10, padding=0):
        super(MixFormer, self).__init__(
            Patch(8,8),
            #Resident(Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            Add(Scale([1.0]),Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            Add(Scale([1.0]),Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            Add(Scale([1.0]),Mixer(48,16,2)),
            Add(Scale([1.0]),Mixer(64,16,1)),
            nn.BatchNorm1d(64),
            nn.Linear(48, 1),
            nn.ReLU(),
            Reshape(-1),
            nn.Linear(64, num_classes),
        )

def Net(**args) :
    return MixFormer(**args)
