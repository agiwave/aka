# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
def GPTAttention(config):
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.in_proj(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * self.scaleDk
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = np.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y

    scaleDk = 1.0/np.sqrt(np.array([config.n_embd//config.n_head]))
    return nn.Module( 
                forward =forward, 
                in_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias),
                out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
                attn_dropout = nn.Dropout(config.dropout),
                resid_dropout = nn.Dropout(config.dropout),
                n_head = config.n_head,
                n_embd = config.n_embd,
                scaleDk = scaleDk,
                bias = np.tril(np.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

def GPTBlock(config):
    return nn.Sequential(
        nn.Resident(nn.Sequential(
            nn.LayerNorm(config.n_embd, bias=config.bias),
            GPTAttention(config)
        )),
        nn.Resident(nn.Sequential(
            nn.LayerNorm(config.n_embd, bias=config.bias),
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )),
    )

def forward(self, idx, targets=None):
    b, t = idx.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = np.arange(0, t, device=idx.device) # shape (t)

    # forward the GPT model itself
    tok_emb = self.t_wte(idx) # token embeddings of shape (b, t, n_embd)
    pos_emb = self.t_wpe(pos) # position embeddings of shape (t, n_embd)
    x = self.t_drop(tok_emb + pos_emb)
    x = self.t_h(x)
    x = self.t_ln_f(x)

    if targets is not None:
        # if we are given some desired targets also calculate the loss
        logits = self.lm_head(x)
        loss = np.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        loss = None
    return logits, loss

def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = np.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = np.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = np.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = np.cat((idx, idx_next), dim=1)
    return idx

def GPT(config):
    assert config.vocab_size is not None
    assert config.block_size is not None

    t_wte = nn.Embedding(config.vocab_size, config.n_embd)
    t_wpe = nn.Embedding(config.block_size, config.n_embd)
    t_drop = nn.Dropout(config.dropout)
    t_h = nn.Sequential(
        *[GPTBlock(config) for _ in range(config.n_layer)]
    )
    # t_h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    t_ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
    lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # with weight tying when using torch.compile() some warnings get generated:
    # "UserWarning: functional_call was passed multiple values for tied weights.
    # This behavior is deprecated and will be an error in future versions"
    # not 100% sure what this is, so far seems to be harmless. TODO investigate
    t_wte.weight = lm_head.weight # https://paperswithcode.com/method/weight-tying
    return nn.Module(
        forward=forward,
        config = config,
        lm_head = lm_head,
        t_wte = t_wte,
        t_wpe = t_wpe,
        t_drop = t_drop,
        t_h = t_h,
        t_ln_f = t_ln_f,
        generate = generate
    )

# --- Example ---
if __name__ == "__main__":
    class PackedDataset:
        def __init__(self, data_dir: str, block_size: int = 1024, filemode: str = "r") -> None:
            super().__init__()

            with open(data_dir, 'r', encoding='utf-8') as f:
                data = f.read()

            # encode with tiktoken gpt2 bpe
            import tiktoken
            encoder = tiktoken.get_encoding("gpt2")
            train_ids = encoder.encode_ordinary(data)

            print(f"train has {len(train_ids):,} tokens")
            self.data = train_ids
            self.block_size = block_size

        def __len__(self) -> int:
            return len(self.data) - self.block_size

        def __getitem__(self, index: int):
            x = np.array(
                self.data[index : index + self.block_size]
            )
            y = np.array(
                self.data[index+1 : index + 1 + self.block_size]
            )
            return x,y
            
    args = nn.Object(
        vocab_size = 50304,
        block_size = 256, # context of up to 256 previous characters

        # baby GPT model :)
        n_layer = 6,
        n_head = 6,
        n_embd = 384,
        dropout = 0.2,
        bias = False, # do we use bias inside LayerNorm and Linear layers?
        lr = 6e-4, # max learning rate
        dataset_path='./data/shakespeare.txt',
        device="cpu",
        batch_size = 12, # if gradient_accumulation_steps > 1, this is the micro-batch size
        epochs=100
    )

    nn.train(
        GPT(args),
        (PackedDataset(args.dataset_path, block_size=args.block_size),None),
        optimizer="Adam",
        optimizer_kwargs={'lr':args.lr},
        show_chart=True,
        batch_size=args.batch_size,
        epochs=args.epochs)
