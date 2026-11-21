import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import config
from models.decoder_timesformer_base import Attention, FeedForward


class NIGExpertHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.EXPERT_HIDDEN),
            nn.SiLU(),
            nn.Linear(config.EXPERT_HIDDEN, 4),
        )
        self.eps = 1e-6

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        gamma = out[..., 0]
        nu = F.softplus(out[..., 1]) + self.eps
        alpha = F.softplus(out[..., 2]) + 1.0 + self.eps
        beta = F.softplus(out[..., 3]) + self.eps
        return gamma, nu, alpha, beta


class _LDBBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_time = nn.LayerNorm(config.EMBED_DIM)
        self.time_attn = Attention(config.EMBED_DIM)
        self.norm_space = nn.LayerNorm(config.EMBED_DIM)
        self.space_attn = Attention(config.EMBED_DIM)
        self.norm_mlp = nn.LayerNorm(config.EMBED_DIM)
        self.mlp = FeedForward(config.EMBED_DIM)

    def forward(self, x):
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


class EPSTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed_dim = config.EMBED_DIM
        self.patch_h = config.PATCH_H
        self.patch_w = config.PATCH_W
        self.num_time_steps = config.NUM_TIME_STEPS
        self.K = config.K
        self.D_k = config.D_K
        self.lambda_ent = config.LAMBDA_ENT

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.EMBED_DIM) for _ in range(config.NUM_FUSION_LAYERS)
        ])
        self.gate_logits = nn.Parameter(torch.zeros(config.NUM_FUSION_LAYERS))

        self.ldb_blocks = nn.ModuleList([
            _LDBBlock() for _ in range(config.LDB_NUM_BLOCKS)
        ])
        self.ldb_norm = nn.LayerNorm(config.EMBED_DIM)

        self.alpha = nn.Parameter(torch.tensor(config.ALPHA_INIT))

        self.prototypes = nn.Parameter(torch.randn(config.K, config.D_K) * 0.02)
        self.W_q = nn.Linear(config.EMBED_DIM, config.D_K, bias=False)
        self.W_k = nn.Linear(config.D_K, config.D_K, bias=False)
        self.log_tau = nn.Parameter(torch.full((config.K,), float(np.log(config.TAU_INIT))))

        self.expert_heads = nn.ModuleList([
            NIGExpertHead() for _ in range(config.K)
        ])

        self._prototypes_initialized = False
        self._last_W = None
        self._last_gamma = None
        self._last_nu = None
        self._last_alpha = None
        self._last_beta = None

    def forward(self, multi_layer_features):
        B = multi_layer_features[0].shape[0]

        gates = F.softmax(self.gate_logits, dim=0)
        F_fused = torch.zeros_like(multi_layer_features[0])
        for i, feat in enumerate(multi_layer_features):
            F_fused = F_fused + gates[i] * self.layer_norms[i](feat)

        x = F_fused
        for block in self.ldb_blocks:
            x = block(x)
        F_transformed = F_fused + self.ldb_norm(x)

        F_mean = F_transformed.mean(dim=1)
        F_std = F_transformed.std(dim=1, unbiased=False)
        F_pooled = F_mean + self.alpha * F_std

        Q = self.W_q(F_pooled)
        K_vec = self.W_k(self.prototypes)
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        attn_logits = torch.matmul(Q, K_vec.T) / (self.D_k ** 0.5)
        attn_logits = attn_logits / tau.unsqueeze(0).unsqueeze(0)
        W = F.softmax(attn_logits, dim=-1)

        gamma_list, nu_list, alpha_list, beta_list = [], [], [], []
        for k in range(self.K):
            g_k, n_k, a_k, b_k = self.expert_heads[k](F_pooled)
            gamma_list.append(g_k)
            nu_list.append(n_k)
            alpha_list.append(a_k)
            beta_list.append(b_k)

        gamma = torch.stack(gamma_list, dim=-1)
        nu = torch.stack(nu_list, dim=-1)
        alpha = torch.stack(alpha_list, dim=-1)
        beta = torch.stack(beta_list, dim=-1)

        y_flat = (W * gamma).sum(dim=-1)

        y_pred = y_flat.reshape(B, self.patch_h, self.patch_w)

        self._last_W = W
        self._last_gamma = gamma
        self._last_nu = nu
        self._last_alpha = alpha
        self._last_beta = beta

        return y_pred

    def compute_entropy_loss(self):
        W = self._last_W
        if W is None:
            return torch.tensor(0.0, device=self.prototypes.device)
        p_mean = W.mean(dim=[0, 1])
        eps = 1e-8
        entropy = -(p_mean * torch.log(p_mean + eps)).sum()
        return -entropy

    @torch.no_grad()
    def initialize_prototypes(self, features, labels, label_mask):
        device = self.prototypes.device
        B, N, D = features.shape
        feats_flat = features.reshape(B, self.patch_h, self.patch_w, D)
        labels_cpu = labels.cpu().numpy()

        collected_feats = []
        collected_labels = []
        for bi in range(B):
            mask = (label_mask[bi] > 0).cpu().numpy()
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            for r, c in zip(rows, cols):
                collected_feats.append(feats_flat[bi, r, c].cpu().numpy())
                collected_labels.append(labels_cpu[bi, r, c])

        if len(collected_feats) < self.K:
            print(f"[EPST] Only {len(collected_feats)} labeled positions, "
                  f"need >= {self.K}. Using random init.")
            return

        collected_feats = np.stack(collected_feats, axis=0)
        collected_labels = np.array(collected_labels)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.K, random_state=config.KMEANS_SEED, n_init=config.KMEANS_N_INIT)
        cluster_ids = kmeans.fit_predict(collected_feats)
        centers = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
        self.prototypes.data = centers

        self._cluster_biomass = []
        for k in range(self.K):
            cl = collected_labels[cluster_ids == k]
            val = float(np.median(cl)) if len(cl) > 0 else float(np.median(collected_labels))
            self._cluster_biomass.append(val)

        self._prototypes_initialized = True
        print(f"[EPST] Prototypes initialized via k-means (K={self.K}). "
              f"Cluster biomass range: [{min(self._cluster_biomass):.1f}, "
              f"{max(self._cluster_biomass):.1f}]")

    def is_initialized(self) -> bool:
        return self._prototypes_initialized

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EPSTMixtureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_reg = config.LAMBDA_REG
        self.lambda_ent = config.LAMBDA_ENT
        self.eps = 1e-8

    def forward(self, y_pred, label_map, label_mask, valid_mask, decoder):
        supervision_mask_2d = (label_mask > 0) & (valid_mask > 0)
        if not supervision_mask_2d.any():
            return {
                "loss": torch.tensor(0.0, device=y_pred.device, requires_grad=True),
                "point_loss": torch.tensor(0.0, device=y_pred.device),
                "reg_loss": torch.tensor(0.0, device=y_pred.device),
                "ent_loss": torch.tensor(0.0, device=y_pred.device),
            }

        B, H, W_spatial = supervision_mask_2d.shape
        supervision_mask_flat = supervision_mask_2d.reshape(B, H * W_spatial)

        W = decoder._last_W[supervision_mask_flat]
        gamma = decoder._last_gamma[supervision_mask_flat]
        nu = decoder._last_nu[supervision_mask_flat]
        alpha = decoder._last_alpha[supervision_mask_flat]
        beta = decoder._last_beta[supervision_mask_flat]
        y_true = label_map[supervision_mask_2d]

        K = decoder.K

        df = (2.0 * alpha).clamp(min=1.1)
        loc = gamma
        scale_sq = beta * (1.0 + nu) / (nu.clamp_min(self.eps) * alpha.clamp_min(self.eps))
        scale = torch.sqrt(scale_sq.clamp_min(self.eps))

        from torch.distributions import StudentT as TorchStudentT
        log_probs_list = []
        for k in range(K):
            dist_k = TorchStudentT(
                df=df[..., k],
                loc=loc[..., k],
                scale=scale[..., k],
            )
            log_probs_list.append(dist_k.log_prob(y_true))
        log_probs = torch.stack(log_probs_list, dim=-1)

        log_joint = torch.log(W.clamp_min(self.eps)) + log_probs
        log_mixture = torch.logsumexp(log_joint, dim=-1)
        nll = -log_mixture.mean()

        with torch.no_grad():
            log_resp = log_joint - log_mixture.unsqueeze(-1)
            responsibility = torch.exp(log_resp).clamp(min=self.eps, max=1.0)
            responsibility = responsibility / responsibility.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        abs_error = torch.abs(y_true.unsqueeze(-1) - loc)
        evidence_penalty = (responsibility * abs_error * (2.0 * nu + alpha)).mean()
        reg_loss = self.lambda_reg * evidence_penalty

        ent_loss = self.lambda_ent * decoder.compute_entropy_loss()

        total_loss = nll + reg_loss + ent_loss

        return {
            "loss": total_loss,
            "point_loss": nll,
            "reg_loss": reg_loss,
            "ent_loss": ent_loss,
        }


class EPSTWarmupLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, label_map, label_mask, valid_mask):
        supervision_mask = (label_mask > 0) & (valid_mask > 0)
        if not supervision_mask.any():
            return {
                "loss": torch.tensor(0.0, device=y_pred.device, requires_grad=True),
                "point_loss": torch.tensor(0.0, device=y_pred.device),
            }
        diff = y_pred - label_map
        point_loss = (diff ** 2 * supervision_mask.float()).sum() / supervision_mask.float().sum().clamp_min(1.0)
        return {
            "loss": point_loss,
            "point_loss": point_loss,
        }
