import torch
import torch.nn as nn

from models import config
from models.mrs_3D import Mrs3D


class MrsOther3DFinetune(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.patch_h = config.PATCH_H
        self.patch_w = config.PATCH_W
        self.num_time_steps = config.NUM_TIME_STEPS
        self.embed_dim = config.EMBED_DIM
        self.encoder = Mrs3D()

    def _encode_timesformer_multi_layer(self, imgs: torch.Tensor, valid_mask: torch.Tensor):
        encoder = self.encoder
        valid_index = encoder._resolve_valid_index_timesformer(imgs, valid_mask)
        tokens = encoder._tokenize_timesformer(imgs)

        B = tokens.shape[0]
        tokens = encoder.patch_embed(tokens.reshape(B, -1, encoder.in_chans))
        tokens = tokens.reshape(B, self.num_time_steps, encoder.num_spatial_patches, self.embed_dim)
        tokens = tokens + encoder.time_pos_embed + encoder.space_pos_embed
        tokens = tokens * valid_index.reshape(B, self.num_time_steps, encoder.num_spatial_patches, 1)

        multi_layer_features = []
        for block_idx, block in enumerate(encoder.blocks):
            tokens = block(tokens)
            if block_idx in config.FUSION_LAYER_INDICES:
                multi_layer_features.append(tokens)
        return multi_layer_features

    def forward(self, imgs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        multi_layer_features = self._encode_timesformer_multi_layer(imgs, valid_mask)
        return self.decoder(multi_layer_features)
