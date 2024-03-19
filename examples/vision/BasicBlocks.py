import aka.nn as nn
import aka.numpy as np

def DownSample2d(input_shape, kernel_size):
    (C,  H,  W ) = input_shape
    (OC, OH, OW) = C*kernel_size*kernel_size, H//kernel_size, W//kernel_size
    (NH, NW) = kernel_size, kernel_size
    assert(NH*OH==H and NW*OW==W)
    return nn.Sequential(
        nn.Reshape(C, NH, OH, NW, OW),
        nn.Permute(1, 2, 4, 3, 5),   # --> [C, NH, NW, OH, OW]
        nn.Reshape(OC, OH, OW)
    )

def UpSample2d(input_shape, kernel_size):
    (C,  H,  W ) = input_shape
    (OC, OH, OW) = C//kernel_size//kernel_size, H*kernel_size, W*kernel_size
    (NH, NW) = kernel_size, kernel_size
    assert(OC*NH*NW==C)
    return nn.Sequential(
        nn.Reshape(OC, NH, NW, H, W),
        nn.Permute(1, 2, 4, 3, 5),   # --> [NH, NW, C, OH, OW]
        nn.Reshape(OC, OH, OW)
    )


