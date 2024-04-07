
import os
os.environ["aka_provider_name"] = "aka.providers.torch"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.nn as nn
import aka.repo as repo
import aka.data

def TrainRoles(roles, *, dataset=None, tokenizer='data/RomeArena', save_dir="data/RomeArena", block_size=512, batch_size=6, lr=1.e-4, dtype=None, show=False, **kwargs):
    # -- dataset --
    if dataset is None:
        dataset = repo.AutoDataset(path='text', data_dir='data/pretrain', split='train')
    elif isinstance(dataset, str):
        dataset = repo.AutoDataset(dataset)
    elif isinstance(dataset, dict):
        dataset = repo.AutoDataset(**dataset)
    # -- Tokenizer --
    if isinstance(tokenizer, str):
        tokenizer = repo.AutoTokenizer(tokenizer)

    # class Tokenizer:
    #     def __init__(self, path):
    #         from sentencepiece import SentencePieceProcessor
    #         self.tokenizer = SentencePieceProcessor('data/Gemma/tokenizer.model')
    #         self.bos_token_id = self.tokenizer.bos_id()
    #         self.eos_token_id = self.tokenizer.eos_id()
    #     def encode(self, s):
    #         return self.tokenizer.encode(s)
    #     def decode(self, s):
    #         return self.tokenizer.decode(s)
    # vocab_size = 256000
    vocab_size = tokenizer.vocab_size
    vocab_size += 0 if not hasattr(tokenizer, 'added_tokens_decoder') else len(tokenizer.added_tokens_decoder)
    vocab_size += 0 if not hasattr(tokenizer, 'added_tokens_encoder') else len(tokenizer.added_tokens_encoder)
    
    # -- Roles --
    players = []
    import importlib
    for role in roles:
        if isinstance(role, str):
            module_name, sub_name = role.split('-')
            module = importlib.import_module(f"examples.text.{module_name}")
            args = getattr(module, f"{module_name}Args")(sub_name)
            players.append(dict(
                name = role,
                args = dict(
                    args,
                    tokenizer = tokenizer,
                    vocab_size = vocab_size
                ),
                persist_filename = None if save_dir is None else f"{save_dir}/{role}.ckt"
            ))
        else:
            players.append(dict(
                args = dict(
                    role,
                    tokenizer = tokenizer,
                    vocab_size = vocab_size,
                ),
                persist_filename = None if save_dir is None else f"{save_dir}/{role['name']}.ckt"
            ))

    # -- Data loader
    dataloader = aka.data.TextStreamingLoader(
                    dataset, 
                    tokenizer=tokenizer, 
                    n_tokens=block_size,
                    batch_size=batch_size,
                    data_mapper=lambda x:x['text'])

    # -- Train --
    def train(role, **kwargs):
        from examples.text.CausalLM import CausalLM
        m = CausalLM(**role['args'])
        if dtype is not None:
            m = m.to(dtype)
        return nn.train(
            m, 
            data_loader=dataloader,
            optimizer="Adam",
            optimizer_kwargs={'lr':lr},
            forward_kwargs={'state':{}},
            persist_filename = role['persist_filename'],
            persist_per_batchs = 50,
            **kwargs)

    # -- Plot --
    losses = [train(r, **kwargs) for r in players]
    if show:
        PlotRoles([r.name for r in players], losses)
    return losses

def PlotRoles(roles, losses):
    from matplotlib import pyplot as plt
    for v in losses:
        plt.plot(v)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend(roles, loc='upper right')
    plt.show()

def RunRoles(roles, prompt, *, tokenizer='data/RomeArena', save_dir='data/RomeArena'):
    # -- Tokenizer --
    if isinstance(tokenizer, str):
        tokenizer = repo.AutoTokenizer(tokenizer)
    vocab_size = tokenizer.vocab_size
    vocab_size += 0 if not hasattr(tokenizer, 'added_tokens_decoder') else len(tokenizer.added_tokens_decoder)
    vocab_size += 0 if not hasattr(tokenizer, 'added_tokens_encoder') else len(tokenizer.added_tokens_encoder)

    # -- Roles --
    players = []
    import importlib
    for role in roles:
        if isinstance(role, str):
            module_name, sub_name = role.split('-')
            module = importlib.import_module(f"examples.text.{module_name}")
            args = getattr(module, f"{module_name}Args")(sub_name)
            players.append(dict(
                name = role,
                args = dict(
                    args,
                    tokenizer = tokenizer,
                    vocab_size = vocab_size,
                    dropout = 0.1,
                    bias = False
                ),
                persist_filename = None if save_dir is None else f"{save_dir}/{role}.ckt"
            ))
        else:
            players.append(dict(
                args = dict(
                    role,
                    tokenizer = tokenizer,
                    vocab_size = vocab_size,
                ),
                persist_filename = None if save_dir is None else f"{save_dir}/{role['name']}.ckt"
            ))

    # -- Run --
    for role in players:
        from CausalLM import CausalLM
        model = CausalLM(**role['args'])
        nn.load_weights(model, role['persist_filename'])
        print(f"{role['name']}:")
        for w in model.generator(prompt):
            print(w, end='')
        print('')

if __name__ == "__main__":
    TrainRoles([
        # 'Gemma-20m', 
        'RomeSet-20m',
        # 'RomeSet-24vdim',
        'RomeSet-Ret',
        # 'RomeSet-32vdim',
        # 'RomeSet-64vdim',
        # 'RomeSet-vbdimpad',
        # 'RomeSet-vbdim',
        # 'RomeSet-novbdim',
    ], lr = 6e-4, epochs=1)