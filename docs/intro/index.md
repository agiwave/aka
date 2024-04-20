# 简介

这只是一个工具，它包装了大量人工智能需要的一些主要函数；

也不是一个工具，因为它几乎没有什么实际的实现代码，仅仅是一些接口代理；

也可以说不仅仅是一个工具，因为用它甚至可以直达当前一些主流GPT模型；

却也算不上一个工具，因为它只是个人用来测试的环境而已，远不及成熟的地步；

## 1、直达主流的GPT模型

### 1.1、Gemma

先从Kaggle网站下载模型数据，比如：https://www.kaggle.com/models/google/gemma/PyTorch/2b-it/2

然后运行如下Python程序：

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

你可能得到的输出结果：
```
Model loaded
(The life meaning is) the purpose or reason for which something exists. It is the driving force behind an organism's actions and decisions.

The life meaning can be determined through various methods, including introspection, observation, and reflection. Introspection involves examining one's own thoughts, feelings, and motivations to understand the underlying purpose of one'
```

### 1.2、Mamba

先从Huggingface下载可用的模型数据，比如：https://huggingface.co/state-spaces/mamba-130m-hf

然后运行如下Python程序：

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

你可能得到的输出结果：

```
Model loaded
（Mamba is) a very popular and popularly used name for a variety of different species of birds. The name Mamba comes from the Latin word mamba, meaning "mamba" or "mamba-like". Mamba is a common name for a variety of birds, including the common cormorant, the common c
```

### 1.3、其它已经或尝试包含的模型

VQVAE、VQGAN、RWKV、RetNet、Hawk、LLAMA等等，当然也包含了一些我自己在实验的代码

## 2、准备环境

（在这儿，php环境的准备，就不做介绍了），命令行执行：

```
>git clone https://github.com/agiwave/aka.git
>cd aka
>pip install -e .
```
然后，就可以在aka\example看到一系列范例代码了。

## 3、第一个范例：判断一个点是否在半径为0.5的圆内

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