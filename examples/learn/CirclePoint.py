import aka.nn as nn
import aka.numpy as np

# 100000 training points(x, y), range: (-1, 1)
train_points = [(np.rand(2) - 0.5) * 2 for _ in range(100000)]
model = nn.Sequential(
    nn.Linear(2, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
)
nn.train(
    model, 
    { 'train': [(x, 1 if np.sum(x**2) > 0.5*0.5 else 0) for x in train_points] },
    loss_metric=nn.CrossEntropyLoss(), 
    batch_size=500, 
    epochs=10)

# batch test points, range: (-1, 1)
test_points = (np.rand(10, 2) - 0.5) * 2
print(test_points)
print(np.softmax(model(test_points), dim=1))
