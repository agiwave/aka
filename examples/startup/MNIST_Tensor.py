import aka.nn as nn
import aka.numpy as np
import aka.data as dataset

def MNIST():
    def __init__(self):
        self.fc1 = nn.Parameter(shape=(784, 576))
        self.bias1 = nn.Parameter(np.zeros(576))
        self.fc2 = nn.Parameter(shape=(576, 10))
        self.bias2 = nn.Parameter(np.zeros(10))
        return self
    def forward(self, x) :
        x = np.flatten(x, 1)
        x = np.matmul(x, self.fc1) + self.bias1
        x = np.relu(x)
        x = np.matmul(x, self.fc2) + self.bias2
        return x
    return __init__(nn.Module(forward=forward))

nn.train(
    MNIST(), 
    dataset.MNIST(),
    loss_metric=nn.CrossEntropyLoss(), 
    batch_size=64, 
    epochs=1, 
    show_chart=True)
