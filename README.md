# Aka -- a toolkit for ai

It's only a proxy to pytorch、transformers、datasets etc. For example: "import aka.numpy" is almost same to "import torch".

## 1、to top GPTs

### 1.1、Gemma

Download gemma data first from：https://www.kaggle.com/models/google/gemma/PyTorch/2b-it/2

run python：

``` python
import sys
sys.path.append('aka')
sys.path.append('aka/examples/text')

from examples.text.Gemma import Gemma

m = Gemma('2b', 'data/gemma/gemma-2b-it.ckpt', 'data/gemma/tokenizer.model')
print('Model loaded')
for w in m.generator("The life meaning is"):
    print(w, end='')
```

The result maybe is：
```
Model loaded
(The life meaning is) the purpose or reason for which something exists. It is the driving force behind an organism's actions and decisions.

The life meaning can be determined through various methods, including introspection, observation, and reflection. Introspection involves examining one's own thoughts, feelings, and motivations to understand the underlying purpose of one'
```

### 1.2、Mamba

Download mamba data first from：https://huggingface.co/state-spaces/mamba-130m-hf

run Python：

``` python
import aka.repo as repo
tokenizer = repo.AutoTokenizer('data/mamba-130m')
cfg = repo.fopen('data/mamba-130m', 'config.json', ftype='json')
args = dict(
    tokenizer = tokenizer,
    vocab_size = 50280, # cfg['vocab_size'],
    latent_dim = cfg['d_model'],
    layers = [
        dict(
            name = 'Mamba',
            hidden_dim = cfg['d_model'] * 2,
            num_heads = cfg['d_model'] * 2,
            dt_rank = cfg['d_model']//16,
            conv_kernel_size = 4,
            conv_bias = True,
            num_states = 16
        )
    ]*cfg['n_layer'],
    bias = False
)

# -- Model --
from CausalLM import CausalLM
mamba = CausalLM(**args)

# -- Weights --
import aka.numpy as np
state = np.load(
    'data/mamba-130m/pytorch_model.bin', mmap=True, weights_only=True,
)

with np.no_grad():
    mamba.embedding.weight.copy_(state['backbone.embedding.weight'])
    mamba.post_norm.weight.copy_(state['backbone.norm_f.weight'])
    for i in range(len(mamba.layers)):
        mamba.layers[i].norm.weight.copy_(state[f'backbone.layers.{i}.norm.weight'])
        mamba.layers[i].layer.A_log.copy_(state[f'backbone.layers.{i}.mixer.A_log'])
        mamba.layers[i].layer.D.copy_(state[f'backbone.layers.{i}.mixer.D'])
        mamba.layers[i].layer.conv1d.weight.copy_(state[f'backbone.layers.{i}.mixer.conv1d.weight'])
        mamba.layers[i].layer.conv1d.bias.copy_(state[f'backbone.layers.{i}.mixer.conv1d.bias'])
        mamba.layers[i].layer.dt_proj.weight.copy_(state[f'backbone.layers.{i}.mixer.dt_proj.weight'])
        mamba.layers[i].layer.dt_proj.bias.copy_(state[f'backbone.layers.{i}.mixer.dt_proj.bias'])
        mamba.layers[i].layer.in_proj.weight.copy_(state[f'backbone.layers.{i}.mixer.in_proj.weight'])
        mamba.layers[i].layer.out_proj.weight.copy_(state[f'backbone.layers.{i}.mixer.out_proj.weight'])
        mamba.layers[i].layer.x_proj.weight.copy_(state[f'backbone.layers.{i}.mixer.x_proj.weight'])

# -- Infer --
print('Model loaded')
for w in mamba.generator("Mamba is"):
    print(w, end='')
```

The result maybe is:

```
Model loaded
（Mamba is) a very popular and popularly used name for a variety of different species of birds. The name Mamba comes from the Latin word mamba, meaning "mamba" or "mamba-like". Mamba is a common name for a variety of birds, including the common cormorant, the common c
```

### 1.3、Some other top models.

VQVAE、VQGAN、RWKV、RetNet、Hawk、LLAMA2 etc. 

## 2、Prepare envirenment.

```
>git clone https://github.com/agiwave/aka.git
>cd aka
>pip install -e .
```
And then, you can find those examples at : aka\example.

## 3、First example maybe is: Whether a point is in a circle(r=0.5).

``` python

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
```