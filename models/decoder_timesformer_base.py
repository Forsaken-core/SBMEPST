import torch
import torch.nn as nn
import torch.nn.functional as F

from models import config


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = int(dim * config.MLP_RATIO)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            GEGLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.heads = config.NUM_HEADS
        self.scale = config.DIM_HEAD ** -0.5
        inner_dim = config.DIM_HEAD * config.NUM_HEADS

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(t.shape[0], t.shape[1], h, -1).permute(0, 2, 1, 3)
            for t in qkv
        )

        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class _ReduceHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        layers = []
        cur_dim = in_dim
        while cur_dim > 16:
            next_dim = max(cur_dim // 2, 16)
            layers.append(nn.Linear(cur_dim, next_dim, bias=True))
            layers.append(nn.GELU())
            cur_dim = next_dim
        layers.append(nn.Linear(cur_dim, config.IN_CHANS, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
