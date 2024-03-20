import aka.nn as nn
import aka.numpy as np

def Topk(n_topk, *, dim=-1):
    def forward(self, x):
        dim = self.dim
        n_topk = self.n_topk
        v, indics = np.topk(x, n_topk, dim=dim)
        v = np.select(v, dim=dim, index=n_topk-1).unsqueeze(dim=dim)
        x = np.where(x<v,float('-inf'), x)
        return np.softmax(x, dim=dim)
    return __init__(nn.Module(forward=forward, n_topk=n_topk, dim=dim))

def KV(kv_size, k_dim, v_dim, gate=True, num_heads=1, act="gelu"):
    def __init__(self):
        assert k_dim % num_heads == 0 and v_dim % num_heads == 0
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.k = nn.Parameter(shape=(num_heads, kv_size, k_dim//num_heads), initializer='xavier_uniform')
        self.v = nn.Parameter(shape=(num_heads, kv_size, v_dim//num_heads), initializer='xavier_uniform')
        self.gate = None if not gate else nn.Parameter(shape=(num_heads, kv_size, k_dim//num_heads), initializer='xavier_uniform')
        self.act = getattr(np,act)
        return self

    def forward(self, x):
        D = x.size(-1)
        assert D == self.k_dim
        input_shape = x.shape
        x = x.view(-1, num_heads, D//num_heads)
        att = np.einsum('bhd,hkd->bhk', x, self.k)
        if self.gate is None:
            att = self.act(att)
        else:
            gate = np.einsum('bhd,hkd->bhk', x, self.gate)
            att = att * gate
        y = np.einsum('bhk,hkd->bhd', att, self.v)
        return y.view(input_shape)

    return __init__(nn.Module(forward=forward))