# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"
import aka.nn as nn
import aka.numpy as np

# -- Self attention --
def SelfAttention(input_shape, latent_dim=0, n_heads=1, atten_dim=0):
    '''Self attention module.

    Args:
        input_shape: Should be (N, D) without B(Batch). N -- Numbers of tokens, D -- Size of tokens
        n_heads: Apply head numbers of attention. D % n_heads must be zero.

    Examples:
        atten = SelfAttention((10, 20), 5)
        input = np.randn(10, 10, 20)
        output = atten(input)
        print(output.size())'''
        
    (N, D) = input_shape
    H = n_heads
    latent_dim = D if latent_dim == 0 else latent_dim
    n_head_latent_dim = latent_dim//H
    atten_dim = n_head_latent_dim if atten_dim == 0 else atten_dim
    scaleDk = 1.0/np.sqrt(np.array([n_head_latent_dim]))
    assert(n_head_latent_dim*H==latent_dim)
    return nn.Sequential(
        # -- Scores[H, N, N] matmul V[H, N, HD] --
        nn.MatMul(
            # -- Score = Softmax((Q matmul V) / sqrtDk)--
            nn.Sequential(
                nn.MatMul(
                    # -- Q --
                    nn.Sequential(
                        nn.Linear(D, H*atten_dim, bias=False),
                        nn.Reshape(N, H, atten_dim),   # [N, D] --> [N, H, HD]
                        nn.Permute(2, 1, 3),    # [N, H, HD] --> [H, N, HD]
                    ),
                    # -- K --
                    nn.Sequential(
                        nn.Linear(D, H*atten_dim, bias=False),
                        nn.Reshape(N, H, atten_dim),   # [N, D] --> [N, H, HD]
                        nn.Permute(2, 3, 1),    # [N, H, HD] --> [H, HD, N]
                    )
                ),
                nn.Scale(scaleDk),
                nn.Softmax(dim=-1)
            ),
            # -- V --
            nn.Sequential(
                nn.Linear(D, latent_dim, bias=False),
                nn.Reshape(N, H, n_head_latent_dim),   # [N, D] --> [N, H, HD]
                nn.Permute(2, 1, 3),   # [N, H, HD] --> [H, N, HD]
            )
        ),
        nn.Permute(2,1,3),           # [H, N, HD] --> [N, H, HD]
        nn.Reshape(N, latent_dim)
    )

# -- Linear attention --
def FFAttention(input_shape, latent_dim=0, n_heads=1):
    '''Linear attention module.

    Args:
        input_shape: Should be (N, D) without B(Batch). N -- Numbers of tokens, D -- Size of tokens
        n_heads: Apply head numbers of attention. D % n_heads must be zero.

    Examples:
        atten = SelfAttention((10, 20), 5)
        input = np.randn(10, 10, 20)
        output = atten(input)
        print(output.size())'''
        
    (N, D) = input_shape
    H = n_heads
    latent_dim = D if latent_dim == 0 else latent_dim
    n_head_latent_dim = latent_dim//H
    assert(n_head_latent_dim*H==latent_dim)
    return nn.Sequential(
        # -- Scores[H, N, N] matmul V[H, N, HD] --
        nn.MatMul(
            # -- Score = Weight(H,N,N)  --
            nn.Parameter(shape=(H,N,N), requires_grad=True, initializer='xavier_uniform'),

            # -- V --
            nn.Sequential(
                nn.Linear(D, latent_dim, bias=False),
                nn.Reshape(N, H, n_head_latent_dim),   # [N, D] --> [N, H, HD]
                nn.Permute(2, 1, 3),   # [N, H, HD] --> [H, N, HD]
            )
        ),
        nn.Permute(2,1,3),           # [H, N, HD] --> [N, H, HD]
        nn.Reshape(N, latent_dim)
    )

# --- Example ---
if __name__ == "__main__":
    atten = SelfAttention((224, 768), 16)
    input = np.randn(50, 224, 768)
    output = atten(input)
    print(output.size())