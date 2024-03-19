# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn

def PatchMixer(num_channels, patch_size, num_patchs):
    C = num_channels
    NPH = num_patchs
    NPW = num_patchs
    PH = patch_size
    PW = patch_size

    # input - [ IC, NPH, PH, NPW, PW]
    return nn.Sequential(     
        nn.Permute(1, 3, 5, 2, 4),  # --[IC, (PH, PW), (NPH, NPW)]
        nn.Reshape(C*PH*PW, NPH*NPW),
        nn.Linear(NPH*NPW, NPH*NPW),
        nn.ReLU(),
        nn.BatchNorm1d(),

        nn.Linear(NPH*NPW, NPH*NPW),
        nn.ReLU(),
        nn.BatchNorm1d(),

        nn.Reshape(C, PH, PW, NPH, NPW),
        nn.Permute(1, 4, 2, 5, 3))

def PatchLinear(num_channels, kernel_size, patch_size, num_patchs):
    IC = num_channels*kernel_size*kernel_size
    C = num_channels
    NPH = num_patchs
    NPW = num_patchs
    PH = patch_size
    PW = patch_size

    # input - [ IC, NPH, PH, NPW, PW]
    return nn.Sequential(
        nn.Reshape(C, NPH*PH, NPW*PW),
        nn.Unfold(kernel_size, stride=1, padding=2),
        nn.Reshape(C, NPH, PH, NPW, PW),

        nn.Permute(3, 5, 2, 4, 1),  # --[(PH, PW), (NPH, NPW), IC]
        nn.Reshape(PH*PW, NPH*NPW, IC),
        nn.MatMul(nn.Parameter(shape=(PH*PW, IC, C), requires_grad=True, initializer="xavier_uniform")),
        nn.ReLU(),
        nn.BatchNorm2d(),
        nn.Reshape(PH, PW, NPH, NPW, C),
        nn.Permute(5, 3, 1, 4, 2),

        nn.Reshape(C, NPH*PH, NPW*PW),
        nn.Unfold(kernel_size, stride=1, padding=2),
        nn.Reshape(C, NPH, PH, NPW, PW),

        nn.Permute(3, 5, 2, 4, 1),  # --[(PH, PW), (NPH, NPW), IC]
        nn.Reshape(PH*PW, NPH*NPW, IC),
        nn.MatMul(nn.Parameter(shape=(PH*PW, IC, C), requires_grad=True, initializer="xavier_uniform")),
        nn.ReLU(),
        nn.BatchNorm2d(),
        nn.Reshape(PH, PW, NPH, NPW, C),
        nn.Permute(5, 3, 1, 4, 2),
    )

from aka.nn import Add, Scale, Resident, Reshape, Parameter
def UnfoldNet(num_classes=100):
    '''
    UnfoldNet:
    1, Conv = Unfold(stride=1) + Linear
    2, Pixel Linear parameters in Patchs are different(without share), Linear Parameters in all patchs are shared.
    
    Result(Compare to ResNet-18):
    1, 10% parameters.
    2, 10x lower speed.
    '''
    return nn.Sequential(
        # Patch
        nn.Input(nn.ImageShape(3,256,256)),
        nn.Reshape(3, 16, 16, 16, 16), # --  [C, NPH, PH, NPW, PW]

        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),
        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),
        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),
        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),
        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),
        Resident(PatchLinear(3, 5, 16, 16)),
        Resident(PatchMixer(3, 16, 16)),

        nn.Permute(2, 4, 1, 3, 5),    # -- [NPH, NPW, C, PH, PW]
        nn.Reshape(16, 16, 768),
        nn.AvgPool2d(16),
        nn.Flatten(),
        nn.Linear(768, num_classes, False),
    )

if __name__ == "__main__":
    import aka.data as datasets
    nn.train(UnfoldNet(), datasets.ImageFolder(root="./datasets/ImageNet", resize=256), batch_size=50, epochs=5)
