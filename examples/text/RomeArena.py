
import os
os.environ["aka_provider_name"] = "aka.providers.torch"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.nn as nn
import aka.repo as repo
import aka.data
from aka.nn import Args

def TrainArena(names, train_args):
    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer('data/mamba-370m-hf')

    # -- Roles --
    roles = [Args(name=name) for name in names]
    import importlib
    for role in roles:
        module_name, sub_name = role.name.split('-')
        module = importlib.import_module(module_name)
        args = getattr(module, module_name+'Args')(sub_name)
        args.tokenizer = tokenizer
        args.vocab_size = 50304
        if not hasattr(args, 'vocab_dim'):
            args.vocab_dim = 64
        args.dropout = 0.1
        args.bias = False
        role.args = args
        role.persist_filename = 'data/RomeArena/'+role.name+".ckt"

    # -- Data loader
    dataset = repo.AutoDataset('text', data_dir='data/text', split='train')
    dataloader = aka.data.TextStreamingLoader(
                    dataset, 
                    tokenizer=tokenizer, 
                    n_tokens=512,
                    batch_size=6,
                    data_mapper=lambda x:x['text'])

    # -- Train --
    def train(role, **kwargs):
        from CausalLM import CausalLM
        return nn.train(
            CausalLM(role.args), 
            data_loader=dataloader,
            optimizer="Adam",
            optimizer_kwargs={'lr':train_args.lr},
            forward_kwargs={'state':{}},
            persist_filename = role.persist_filename,
            epochs=train_args.epochs)

    # -- Plot --
    m_losses = [train(r) for r in roles]
    from matplotlib import pyplot as plt
    for v in m_losses:
        plt.plot(v)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend([r.name for r in roles], loc='upper right')
    plt.show()

def RunArena(names, prompt):
    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer('data/mamba-370m-hf')

    # -- Roles --
    roles = [Args(name=name) for name in names]
    import importlib
    for role in roles:
        module_name, sub_name = role.name.split('-')
        module = importlib.import_module(module_name)
        args = getattr(module, module_name+'Args')(sub_name)
        args.tokenizer = tokenizer
        args.vocab_size = 50304
        if not hasattr(args, 'vocab_dim'):
            args.vocab_dim = 64
        args.dropout = 0.1
        args.bias = False
        role.args = args
        role.persist_filename = 'data/RomeArena/'+role.name+".ckt"

    # -- Run --
    for role in roles:
        from CausalLM import CausalLM
        model = CausalLM(role.args)
        nn.load_weights(model, role.persist_filename)
        print(role.name + ":")
        for w in model.generator(prompt):
            print(w, end='')
        print('')

if __name__ == "__main__":
    TrainArena([
        # 'Gemma-20m', 
        'RomeSet-20m',
        # 'RomeSet-24vdim',
        'RomeSet-Ret',
        # 'RomeSet-32vdim',
        # 'RomeSet-64vdim',
        # 'RomeSet-vbdimpad',
        # 'RomeSet-vbdim',
        # 'RomeSet-novbdim',
        ], Args(lr = 6e-4, epochs=1)
    )