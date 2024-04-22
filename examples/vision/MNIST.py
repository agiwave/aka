import aka.nn as nn
import aka.data as dataset

def MNIST():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 576),
        nn.ReLU(),
        nn.Linear(576, 10)
    )

nn.train(
    MNIST(), 
    dataset.MNIST(),
    loss_metric=nn.CrossEntropyLoss(), 
    batch_size=64, 
    epochs=1, 
    show_chart=True)
