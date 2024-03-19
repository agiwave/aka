
import os
os.environ["aka_provider_name"] = "aka.providers.torch"

import aka.nn as nn
import aka.data as datasets

'''
数据集尺寸
    MNIST: 28 * 28
    FashionMNIST: 28 * 28
    CIFAR10、CIFAR100: 3 * 32 * 32

网络输入尺寸
    LeNet:      32 * 32 * ?
    AlexNet:    224 * 224 * ? 
                227 * 227 * 3, padding=0
                65 * 65 * ?, padding=0, stride=1
                64 * 64 * ?, padding=2, stride=1
    VGG:      > 224 * 224 * ?
    ResNet:   > 224 * 224 * ?
    DenseNet: > 224 * 224 * ?
    ENet:     > 224 * 224 * ?

'''

def Net(num_channels=1, num_classes=10):
    return nn.Sequential(
        nn.Input(nn.ImageShape(1,28,28)),
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )

nn.train(Net(), datasets.MNIST(root='data'), batch_size=300, epochs=5)
