
# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np
import aka.repo as repo
import aka.data
from CausalLM import CausalLM

if __name__ == "__main__":
    # encode with tiktoken gpt2 bpe
    tokenizer = repo.AutoTokenizer('data/mamba-370m-hf')
    dataset = repo.AutoDataset('text', data_files=[
        'data/wiki/wiki.txt',
        'data/shakespeare/train.txt'
    ])

    args = nn.Args(
        tokenizer = tokenizer,
        vocab_size = 50257,
        vocab_dim = 64,
        latent_dim = 768,
        dropout = 0.0,
        bias = False, # do we use bias inside LayerNorm and Linear layers?

        layers = ['Attention', 'MLP']*6,
        mlp_args = nn.Args(
            kv_size = 768*4,
            kv_gate = False,
        ),
        attn_args = nn.Args(
            window_size = 256,
            num_heads = 12,
            num_kv_groups = 12,
            rotary_embedding = True,
        ),

        # -- Train args --
        lr = 6e-4, # max learning rate
        batch_size = 12, # if gradient_accumulation_steps > 1, this is the micro-batch size
        epochs=5
    )
    dataloader = aka.data.TextStreamingLoader(
                    dataset['train'], 
                    tokenizer=tokenizer, 
                    n_tokens=512, 
                    batch_size=args.batch_size,
                    data_mapper=lambda x:x['text'])

    model = CausalLM(args)
    nn.train(
        model,
        data_loader=dataloader,
        optimizer="Adam",
        optimizer_kwargs={'lr':args.lr},
        show_chart=True,
        persist_filename='GPT2.pkl',
        batch_size=args.batch_size,
        epochs=args.epochs)
    print(model.generate("you know Caius Marcius is"))
