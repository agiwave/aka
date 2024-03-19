# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.repo as repo
import aka.data

if __name__ == "__main__":
    tokenizer = repo.AutoTokenizer('data/mamba-370m-hf')
    dataset = repo.AutoDataset('text', data_files=[
        'data/wiki/wiki.txt',
        'data/shakespeare/train.txt'
    ])

    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs: setattr(self, key, kwargs[key])

    args = Args(
        tokenizer = tokenizer,
        vocab_size = 50304,
        vocab_dim = 64,
        latent_dim = 384,
        dropout = 0.0,
        bias = False, # do we use bias inside LayerNorm and Linear layers?

        # -- Layer args --
        layers = ['Attention', 'MLP']*12,
        mlp_args = Args(
            qk_dim = 64,
            kv_size = 384*4,
            kv_gate = True,
        ),
        attn_args = Args(
            window_size = 256,
            num_heads = 6,
            num_kv_groups = 6,
            rotary_embedding = True,
        ),

        # -- Train args --
        lr = 6e-4,
        batch_size = 12,
        epochs=21
    )
    dataloader = aka.data.StreamingLoader(
                    dataset['train'], 
                    tokenizer=tokenizer, 
                    n_tokens=512, 
                    batch_size=args.batch_size,
                    data_mapper=lambda x:x['text'])

    from CausalLM import CausalLM
    model = CausalLM(args)
    nn.train(
        model,
        data_loader=dataloader,
        optimizer="Adam",
        optimizer_kwargs={'lr':args.lr},
        show_chart=True,
        persist_filename='demoGPT12.pkl',
        batch_size=args.batch_size,
        epochs=args.epochs)
    print(model.generate("you know Caius Marcius is"))
