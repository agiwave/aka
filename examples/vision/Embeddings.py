
import aka.numpy as np


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size)
    grid_w = np.arange(grid_size)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.cat([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.cat([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2)
    omega = omega / embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.cat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# 1d绝对sin_cos编码
def create_1d_absolute_sin_cos_embedding(len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = np.zeros(len, dim)
    # i矩阵
    i_matrix = np.arange(dim//2)
    i_matrix = i_matrix / (dim / 2)
    i_matrix = np.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    # pos矩阵
    pos_vec = np.arange(len) * 1.0
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = np.cos(out)
    emb_sin = np.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb

def create_2d_absolute_sin_cos_embedding(h, w, dim):
    # 奇数列和偶数列sin_cos，还有h和w方向，因此维度是4的倍数
    assert dim % 4 == 0, "wrong dimension"

    pos_emb = np.zeros([h*w, dim])
    m1, m2 = np.meshgrid(np.arange(h), np.arange(w))
    # [2, h, 2]
    coords = np.stack([m1, m2], dim=0)
    # 高度方向的emb
    h_emb = create_1d_absolute_sin_cos_embedding(np.flatten(coords[0]).numel(), dim // 2)
    # 宽度方向的emb
    w_emb = create_1d_absolute_sin_cos_embedding(np.flatten(coords[1]).numel(), dim // 2)
    # 拼接起来
    pos_emb[:, :dim//2] = h_emb
    pos_emb[:, dim//2:] = w_emb
    return pos_emb

if __name__ == '__main__':
    print(create_1d_absolute_sin_cos_embedding(4, 4))
    print(create_2d_absolute_sin_cos_embedding(2, 2, 4))
    print(get_2d_sincos_pos_embed(4,2))

