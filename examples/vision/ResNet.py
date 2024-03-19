# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn

# 这是残差网络中的basicblock
def BasicBlock(inplanes, planes, stride=1, downsample=None):
    layers = [
        # Conv1
        nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                padding=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(),

        # Conv2
        nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                padding=1, bias=False),
        nn.BatchNorm2d(planes),
    ]

    if(downsample != None):
        return nn.Sequential(
            nn.Add(
                downsample,
                nn.Sequential(*layers)
            ),
            nn.ReLU()
        )

    return nn.Sequential(
        nn.Resident(
            nn.Sequential(*layers)
        ),
        nn.ReLU()
    )

def Bottleneck(inplanes, planes, stride=1, downsample=None):
    layers = [
        # Conv1
        nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(),

        # Conv2
        nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(planes * 4),
        nn.ReLU(),

        # Conv3
        nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(planes * 4),
        nn.ReLU(),
    ]

    if(downsample != None):
        return nn.Sequential(
            nn.Add(
                downsample,
                nn.Sequential(*layers)
            ),
            nn.ReLU()
        )

    return nn.Sequential(
        nn.Resident(
            nn.Sequential(*layers)
        ),
        nn.ReLU()
    )

# 每一种架构下都有训练好的可以用的参数文件
# model_urls = {
#     'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
# }

model_blocks = {
    'resnet18': BasicBlock,
    'resnet34': BasicBlock,
    'resnet50': Bottleneck,
    'resnet101': Bottleneck,
    'resnet152': Bottleneck,
}

model_block_expansions = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4,
    'resnet152': 4,
}

model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

def _make_layer(block, expansion, inplanes, planes, blocks, stride=1):
    downsample = None
    #-------------------------------------------------------------------#
    #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
    #-------------------------------------------------------------------#
    if stride != 1 or inplanes != planes * expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion,kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * expansion),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * expansion
    for i in range(1, blocks):
        layers.append(block(planes, planes))
        inplanes = planes
    return nn.Sequential(*layers)

def ResNet(num_channels=3, num_classes=10, num_layers=18, kernel_size=7, stride=2, padding=3) :
    name = "resnet"+str(num_layers)
    if(model_blocks[name] == None):
        name="resnet18"

    block = model_blocks[name]
    layers = model_layers[name]
    expansion = model_block_expansions[name]
    return nn.Sequential(        
        # -- Input --
        nn.Input(nn.ImageShape(3, 224, 224)),

        # -- Prepare
        nn.Conv2d(num_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),

        # -- Conv --
        # 150,150,64 -> 150,150,256
        _make_layer(block, expansion, 64, 64, layers[0]),
        # 150,150,256 -> 75,75,512
        _make_layer(block, expansion, 64*expansion, 128, layers[1], stride=2),
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        _make_layer(block, expansion, 128*expansion, 256, layers[2], stride=2),
        # self.layer4被用在classifier模型中
        _make_layer(block, expansion, 256*expansion, 512, layers[3], stride=2),

        # -- Classifier --
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(512 * expansion, num_classes),
    )

if __name__ == "__main__":
    import aka.data as datasets
    nn.train(
        ResNet(num_classes=100), datasets.ImageFolder(root="./datasets/ImageNet", resize=224),
        loss_metric=nn.CrossEntropyLoss(), 
        batch_size=64, epochs=5)


