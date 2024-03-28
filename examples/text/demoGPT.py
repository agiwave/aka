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
    mlp_args = nn.Object(
        name = 'MLP',
        qk_dim = 64,
        kv_size = 384*4,
        kv_gate = True,
    )
    attn_args = nn.Object(
        name = 'Attention',
        window_size = 256,
        num_heads = 6,
        num_kv_groups = 6,
        rotary_embedding = True,
    )
    args = nn.Object(
        tokenizer = tokenizer,
        vocab_size = 50304,
        vocab_dim = 64,
        latent_dim = 384,
        dropout = 0.0,
        bias = False, # do we use bias inside LayerNorm and Linear layers?

        # -- Layer args --
        layers = [attn_args, mlp_args]*12,

        # -- Train args --
        lr = 6e-4,
        batch_size = 12,
        epochs=21
    )
    dataloader = aka.data.TextStreamingLoader(
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
