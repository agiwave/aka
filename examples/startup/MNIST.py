import aka.nn as nn
import aka.data as dataset

model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 576),
        nn.ReLU(),
        nn.Linear(576, 10)
    )

nn.train(
    model, 
    dataset.MNIST(),
    loss_metric=nn.CrossEntropyLoss(), 
    batch_size=64, 
    epochs=1, 
    show_chart=True)
