import math
import aka.nn as nn
import aka.numpy as np

def MetaLayer(name, args):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def forward(self, x, **kwargs):
        y = self.norm(x)
        y = self.layer(y, **kwargs)
        if isinstance(y, tuple):
            y, loss = y
            return x+y, loss
        else:
            return x+y, None

    import importlib
    module = importlib.import_module(name)
    short_name = name.split('./\\')[-1]
    m = getattr(module, short_name+"Block", None)
    assert m is not None, f"Unknown layer:{name}"
    return nn.Module(forward = forward,norm = nn.RMSNorm(args.latent_dim),layer = m(args))

def CausalLM(args):
    '''
    Causal Language Model.
    '''
    def __init__(self, args):
        in_proj, out_proj = None, None
        vocab_dim = getattr(args, 'vocab_dim', args.latent_dim)
        if vocab_dim != args.latent_dim:
            in_proj = nn.Linear(vocab_dim, args.latent_dim, bias=args.bias)
            out_proj = nn.Linear(args.latent_dim, vocab_dim, bias=args.bias)

        pad_x = getattr(args, 'pad_x', False)
        lm_head = getattr(args, 'lm_head', False)
        make_layer = MetaLayer if not hasattr(args, 'MetaLayer') else args.MetaLayer

        prev_norm = getattr(args, 'prev_norm', None)
        if prev_norm is not None:
            match prev_norm:
                case 'gemma':
                    from Gemma import GemmaEmbNorm
                    prev_norm = GemmaEmbNorm()
                case _:
                    prev_norm = nn.RMSNorm(args.latent_dim)

        embedding_scale = getattr(args,'embedding_scale',False)
        self.tokenizer = args.tokenizer
        self.vocab_dim = vocab_dim
        self.latent_dim = args.latent_dim
        self.pad_x = pad_x
        self.embedding_scale = (None if not embedding_scale else math.sqrt(vocab_dim))
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=vocab_dim)
        self.layers = nn.ModuleList([make_layer(key, args) for key in args.layers])
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.lm_head = None if not lm_head else nn.Linear(vocab_dim, args.vocab_size,bias=False)
        self.prev_norm = prev_norm
        self.post_norm = nn.RMSNorm(args.latent_dim)
        self.gctx = {}
        return self

    def forward(self, inputs, targets=None, state=None):
        # -- Embedding and layers
        x = self.embedding(inputs)

        # -- vocab_dim --> latent_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.latent_dim-self.vocab_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.in_proj(x)
        if self.embedding_scale is not None:    # RetNet, nonsense :(. 
            x = x * self.embedding_scale

        # -- layers --
        if self.prev_norm is not None:
            x = self.prev_norm(x)
        gctx = self.gctx
        if(state is not None):
            layer_states = state.get('layer_states', None)
            if layer_states is None:
                layer_states = [{} for _ in self.layers]
                state['layer_states'] = layer_states

        layer_losses = []
        for i in range(len(self.layers)):
            l_state = None if state is None else layer_states[i]
            x, loss = self.layers[i](x, targets=targets, gctx=gctx, state=l_state)
            if loss is not None:
                layer_losses.append(loss)

        if self.post_norm is not None:
            x = self.post_norm(x)

        # -- latent_dim --> vocab_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.vocab_dim-self.latent_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.out_proj(x)

        # -- vocab_dim --> logits
        if self.lm_head is not None:
            y = self.lm_head(x)    # -- LLaMA vs embedding.weight ? --
        else:
            y = np.einsum('bld,nd->bln', x, self.embedding.weight)

        # -- logits --> output
        if(targets is not None):
            loss = np.cross_entropy(y.view(-1, y.size(-1)), targets.reshape(-1), ignore_index=-1)
            vocab_max = np.max(self.embedding.weight, dim=1)[0]-1.
            vocab_min = np.min(self.embedding.weight, dim=1)[0]
            loss += np.mean(vocab_max**2)+np.mean(vocab_min**2)
            if len(layer_losses) > 0:
                loss += np.sum(np.stack(layer_losses, dim=-1)) / len(layer_losses)
            return y, loss
        else:
            return y

    def generator(self, prompts: str, max_length : int = 64):
        prompt_tokens = [self.tokenizer.bos_token_id]+self.tokenizer.encode(prompts) # [self.tokenizer.bos_token_id]+
        if hasattr(self, 'eval'):
            self.eval()

        with np.no_grad():
            state = {}
            cache = []
            input_token_ids = np.array([prompt_tokens])
            for _ in range(max_length):
                outputs = self(input_token_ids, state=state)
                input_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                cache = cache + input_token_ids[0].tolist()
                if self.tokenizer.eos_token_id in input_token_ids:
                    break

                word = self.tokenizer.decode(cache)
                word_token_ids = self.tokenizer.encode(word)
                if cache[-1] == word_token_ids[-1]:
                    cache = []
                    yield word

            if len(cache)>0:
                yield self.tokenizer.decode(cache)

    def generate(self, prompts : str, max_length : int = 64):
        response = ''
        for w in self.generator(prompts,max_length):
            response += w
        return response
        
    return __init__(nn.Module(forward = forward, generate = generate, generator=generator),args)

if __name__ == "__main__":
    # encode with tiktoken gpt2 bpe
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('data/mamba-370m-hf')
    class DataLoader:
        def __init__(self, data_dir: str, block_size: int = 1024, filemode: str = "r", batch_size=12) -> None:

            with open(data_dir, 'r', encoding='utf-8') as f:
                data = f.read()

            train_ids = tokenizer.encode(data)
            print(f"train has {len(train_ids):,} tokens")

            batch_length = len(train_ids) // batch_size
            batchs = [
                (
                    np.cat([
                        np.array(train_ids[
                            i_row*batch_length+i_col*block_size : i_row*batch_length+(i_col+1)*block_size
                        ]).unsqueeze(0)
                        for i_row in range(batch_size)]
                    ),
                    [True for _ in range(batch_size)]
                )
                for i_col in range(batch_length // block_size)
            ]
            self.batchs = batchs

        def __len__(self) -> int:
            return len(self.batchs)

        def __iter__(self):
            return iter(self.batchs)
            
    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs: setattr(self, key, kwargs[key])

    def train(persist_filename=None, **kwargs):
        args = Args(
            tokenizer = tokenizer,
            vocab_size = 50304,
            vocab_dim = 64,
            block_size = 256,
            latent_dim = 384,
            dropout = 0.2,
            bias = False, # do we use bias inside LayerNorm and Linear layers?

            layers = ['Attention', 'MLP']*6,
            mlp_args = Args(
                kv_size = 384*4,
                kv_gate = True,
                qk_dim = 384,
                hidden_dim = 384
            ),
            attn_args = Args(
                size = 256,
                qk_dim = 384,
                hidden_dim = 384,
                num_heads = 6,
                num_kv_groups = 6,
                rotary_embedding = True,
            ),
            mamba_args = Args(
                hidden_dim = 160,
                dt_rank = 24, # args.latent_dim // 16
                conv_kernel_size = 4,
                conv_bias = True,
                d_state = 16
            ),

            # -- Train args --
            lr = 6e-4, # max learning rate
            dataset_path='./data/shakespeare.txt',
            batch_size = 24,
            epochs=1
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        return nn.train(
            CausalLM(args), 
            data_loader=DataLoader(args.dataset_path, block_size=args.block_size//2, batch_size=args.batch_size),
            optimizer="Adam",
            optimizer_kwargs={'lr':args.lr},
            forward_kwargs={'state':{}},
            persist_filename = persist_filename,
            epochs=args.epochs)

    trains = {
        'att' : train(layers=['Attention', 'MLP']*6, vocab_dim=64),
        'attv' : train(layers=['Attention', 'MLP']*6, vocab_dim=384),
        # 'att' : train(layers=['Attention', 'MLP']*6,),
        # 'catt' : train(layers=['Attention', 'MLP']*6),
        # 'attg' : train(layers=['Attention', 'MLP']*6, mlp_gate=True),
        # 'cattg' : train(layers=['Attention', 'MLP']*6, mlp_gate=True)
    }

    from matplotlib import pyplot as plt
    for _, v in trains.items():
        plt.plot(v)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend([k for k in trains], loc='upper right')
    plt.show()



# if len(prompt_tokens) > 1:
#     self(np.array([prompt_tokens[:-1]]), state=state)
# input_token_ids = np.array([prompt_tokens])
# for _ in range(max_length):
#     outputs = self(input_token_ids, state=state)
#     output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
#     cache = cache + output_token_ids[0].tolist()
#     if self.tokenizer.eos_token_id in input_token_ids:
#         break

#     word = self.tokenizer.decode(cache)
#     word_token_ids = self.tokenizer.encode(word)
#     if cache == word_token_ids:
#         cache = []
#         yield word

#     input_token_ids = np.cat([input_token_ids, output_token_ids], dim=1)
