import numpy as np
import torch
import torch.nn as nn

from models import config
from models.decoder_timesformer_base import Attention, FeedForward, _ReduceHead
from models.pos_embed import get_timesformer_pos_embeds


class VectorTokenEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(config.IN_CHANS, config.EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimeSFormerBaseBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        self.time_attn = Attention(dim)
        self.norm_space = nn.LayerNorm(dim)
        self.space_attn = Attention(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_time = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_time = x_time + self.time_attn(self.norm_time(x_time))
        x = x_time.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x_space = x.reshape(B * T, N, D)
        x_space = x_space + self.space_attn(self.norm_space(x_space))
        x = x_space.reshape(B, T, N, D)
        x_flat = x.reshape(B * T * N, D)
        x_flat = x_flat + self.mlp(self.norm_mlp(x_flat))
        return x_flat.reshape(B, T, N, D)


class Mrs3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chans = config.IN_CHANS
        self.patch_h = config.PATCH_H
        self.patch_w = config.PATCH_W
        self.num_time_steps = config.NUM_TIME_STEPS
        self.embed_dim = config.EMBED_DIM
        self.num_spatial_patches = self.patch_h * self.patch_w
        self.num_patches = self.num_spatial_patches * self.num_time_steps

        self.patch_embed = VectorTokenEmbed()
        self.decoder_pred = _ReduceHead(config.DECODER_EMBED_DIM)
        self.decoder_embed = nn.Linear(self.embed_dim, config.DECODER_EMBED_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.DECODER_EMBED_DIM))
        self.decoder_norm = nn.LayerNorm(config.DECODER_EMBED_DIM)

        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_time_steps, 1, self.embed_dim), requires_grad=False
        )
        self.space_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_spatial_patches, self.embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [TimeSFormerBaseBlock(self.embed_dim) for _ in range(config.DEPTH)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.decoder_time_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_time_steps, 1, config.DECODER_EMBED_DIM), requires_grad=False
        )
        self.decoder_space_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_spatial_patches, config.DECODER_EMBED_DIM), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [TimeSFormerBaseBlock(config.DECODER_EMBED_DIM) for _ in range(config.DECODER_DEPTH)]
        )

        self.initialize_weights()

    def initialize_weights(self):
        time_pos, space_pos, _ = get_timesformer_pos_embeds(
            self.embed_dim, self.patch_h, self.patch_w, self.num_time_steps
        )
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos).float().reshape(1, self.num_time_steps, 1, -1))
        self.space_pos_embed.data.copy_(
            torch.from_numpy(space_pos).float().reshape(1, 1, self.num_spatial_patches, -1)
        )

        dec_time_pos, dec_space_pos, _ = get_timesformer_pos_embeds(
            config.DECODER_EMBED_DIM, self.patch_h, self.patch_w, self.num_time_steps
        )
        self.decoder_time_pos_embed.data.copy_(
            torch.from_numpy(dec_time_pos).float().reshape(1, self.num_time_steps, 1, -1)
        )
        self.decoder_space_pos_embed.data.copy_(
            torch.from_numpy(dec_space_pos).float().reshape(1, 1, self.num_spatial_patches, -1)
        )
        self.decoder_pred.init_weights()

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _tokenize_timesformer(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return (
            x.permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(batch_size, self.num_time_steps, self.num_spatial_patches, self.in_chans)
        )

    def _target_timesformer(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokenize_timesformer(x).reshape(x.shape[0], -1, self.in_chans)

    def _resolve_valid_index_timesformer(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones((x.shape[0], self.num_patches), device=x.device, dtype=x.dtype)
        spatial_valid = (valid_mask.reshape(valid_mask.shape[0], -1) > 0).float()
        spatial_valid = spatial_valid.unsqueeze(1).repeat(1, self.num_time_steps, 1)
        return spatial_valid.reshape(valid_mask.shape[0], -1)

    @staticmethod
    def random_masking_zero_fill(x: torch.Tensor):
        batch_size, length, dim = x.shape
        len_mask = int(length * config.MASK_RATIO)
        len_mask = max(1, min(length, len_mask))
        noise = torch.rand(batch_size, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros([batch_size, length], device=x.device)
        mask[:, :len_mask] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x * (1.0 - mask.unsqueeze(-1))
        return x_masked, mask

    def forward_encoder(self, x: torch.Tensor, valid_mask: torch.Tensor = None):
        valid_index = self._resolve_valid_index_timesformer(x, valid_mask)
        x = self._tokenize_timesformer(x)
        batch_size = x.shape[0]

        x = self.patch_embed(x.reshape(batch_size, -1, self.in_chans))
        x = x.reshape(batch_size, self.num_time_steps, self.num_spatial_patches, self.embed_dim)
        x = x + self.time_pos_embed + self.space_pos_embed
        x = x * valid_index.reshape(batch_size, self.num_time_steps, self.num_spatial_patches, 1)
        x, mask = self.random_masking_zero_fill(
            x.reshape(batch_size, self.num_patches, self.embed_dim)
        )
        x = x.reshape(batch_size, self.num_time_steps, self.num_spatial_patches, self.embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, None, valid_index

    def forward_decoder(self, latent: torch.Tensor, mask: torch.Tensor):
        batch_size = latent.shape[0]
        x = self.decoder_embed(latent)
        mask = mask.reshape(batch_size, self.num_time_steps, self.num_spatial_patches, 1)
        x = x * (1.0 - mask) + self.mask_token * mask
        x = x + self.decoder_time_pos_embed + self.decoder_space_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x.reshape(batch_size, self.num_patches, self.in_chans)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, valid_index: torch.Tensor):
        target = self._target_timesformer(imgs)
        mean = target.mean(dim=1, keepdim=True)
        var = target.var(dim=1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        effective_mask = mask * valid_index
        return (loss * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)

    def forward(self, imgs: torch.Tensor, valid_mask: torch.Tensor = None):
        latent, mask, _ids_restore, valid_index = self.forward_encoder(imgs, valid_mask)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(imgs, pred, mask, valid_index)
        return loss, pred, mask
