import numpy as np


def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    pos = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        cls_embed = np.zeros([1, embed_dim], dtype=np.float32)
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_from_hw(embed_dim, grid_h, grid_w):
    grid_h_vec = np.arange(grid_h, dtype=np.float32)
    grid_w_vec = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_h_vec, grid_w_vec, indexing="ij")
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)


def get_timesformer_pos_embeds(embed_dim, patch_h, patch_w, num_time_steps):
    num_spatial = patch_h * patch_w
    time_pos = get_1d_sincos_pos_embed(embed_dim, num_time_steps, cls_token=False)
    space_pos = get_2d_sincos_pos_embed_from_hw(embed_dim, patch_h, patch_w)
    return time_pos, space_pos, num_spatial
