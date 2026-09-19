"""Student policy: FMT-style fusion of noisy state (24) + wrench history
(32x6) + third-person & wrist RGB (256x256 x2) -> 7-dim action.

Fusion core (WrenchTokenizer, BidirectionalCrossAttention, AttentionPool1d)
is ported from force_aware_fmt.py (the ManipForce FMT design), stripped of
the OpenVLA-OFT-specific bridging: no 4096-dim LLM vision-token interface,
no vision_up/wrench_up projections back into an LLM token stream, no
zero-init "identity at init" residual trick (that existed to make an
untrained bridge behave like the pretrained VLA at step 0 -- irrelevant
here, this network trains from scratch as a standalone BC policy).

Design ref: Downloads/PHASE2_IMPLEMENTATION_SPEC_2026-07-05.md section 3.2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision


@dataclass
class StudentFMTConfig:
    state_dim: int = 24
    wrench_dim: int = 6
    wrench_horizon: int = 32
    action_dim: int = 7
    proj_dim: int = 256
    num_heads: int = 4
    num_vision_tokens_pooled: int = 4
    num_wrench_tokens_pooled: int = 4
    dropout: float = 0.0
    pretrained_backbone: bool = True


class WrenchTokenizer(nn.Module):
    """6D wrench sequence -> token sequence (ported verbatim from force_aware_fmt.py).

    Input  (B, T, 6) z-scored wrench history. Output (B, T/4, proj_dim).
    """

    def __init__(self, cfg: StudentFMTConfig):
        super().__init__()
        d = cfg.proj_dim
        self.net = nn.Sequential(
            nn.Conv1d(cfg.wrench_dim, d // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d // 2, d, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=1, padding=1),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, wrench: torch.Tensor) -> torch.Tensor:
        x = self.net(wrench.transpose(1, 2)).transpose(1, 2)  # (B, T', D)
        return self.norm(x)


class BidirectionalCrossAttention(nn.Module):
    """vision<->wrench cross-attention ('parallel' variant, ported verbatim)."""

    def __init__(self, cfg: StudentFMTConfig):
        super().__init__()
        d, h = cfg.proj_dim, cfg.num_heads
        self.vision_attends_wrench = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.wrench_attends_vision = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.vision_norm1 = nn.LayerNorm(d)
        self.wrench_norm1 = nn.LayerNorm(d)
        self.vision_ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.wrench_ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.vision_norm2 = nn.LayerNorm(d)
        self.wrench_norm2 = nn.LayerNorm(d)

    def forward(self, vision_tok: torch.Tensor, wrench_tok: torch.Tensor):
        # parallel: both directions read the *pre-update* other modality
        v_q = self.vision_norm1(vision_tok)
        w_q = self.wrench_norm1(wrench_tok)
        v_upd, _ = self.vision_attends_wrench(v_q, w_q, w_q, need_weights=False)
        w_upd, _ = self.wrench_attends_vision(w_q, v_q, v_q, need_weights=False)
        vision_tok = vision_tok + v_upd
        wrench_tok = wrench_tok + w_upd
        vision_tok = vision_tok + self.vision_ffn(self.vision_norm2(vision_tok))
        wrench_tok = wrench_tok + self.wrench_ffn(self.wrench_norm2(wrench_tok))
        return vision_tok, wrench_tok


class AttentionPool1d(nn.Module):
    """Pool a token sequence to a fixed number of query tokens (ported verbatim)."""

    def __init__(self, cfg: StudentFMTConfig, num_queries: int):
        super().__init__()
        d = cfg.proj_dim
        self.queries = nn.Parameter(torch.zeros(1, num_queries, d))
        nn.init.normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled)


class CameraBackbone(nn.Module):
    """ResNet18 spatial-feature extractor: (B,256,256,3) uint8 -> (B, 64, proj_dim).

    Uses the layer4 feature map (8x8x512 for a 256x256 input) as a spatial
    token grid, NOT the globally-pooled 512-d embedding -- cross-attention
    with the wrench tokens needs a spatial token sequence, not one vector.
    """

    def __init__(self, cfg: StudentFMTConfig):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained_backbone else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj = nn.Linear(512, cfg.proj_dim)
        # ImageNet normalization. Input convention: uint8 HWC RGB in [0,255].
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """rgb_uint8: (B, H, W, 3) uint8 -> tokens (B, 64, proj_dim)."""
        x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, 8, 8) for a 256x256 input
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 64, 512)
        return self.proj(x)  # (B, 64, proj_dim)


class StudentFMTPolicy(nn.Module):
    """noisy state (24) + wrench history (32x6) + 2x RGB (256x256) -> action (7).

    forward(state, wrench, tp_rgb, wrist_rgb) -> action in [-1, 1]^7.
    exec=1 closed loop: call once per policy step, no action chunking
    (chunked execution was shown to hurt mm-level precision in the prior
    force-aware-VLA project -- see project_force_vla_residual_rl memory).
    """

    def __init__(self, cfg: StudentFMTConfig = StudentFMTConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.proj_dim

        # Separate weights per camera (not shared) -- third-person and wrist
        # views have very different statistics/framing.
        self.tp_backbone = CameraBackbone(cfg)
        self.wrist_backbone = CameraBackbone(cfg)

        self.wrench_tokenizer = WrenchTokenizer(cfg)
        t_tokens = cfg.wrench_horizon // 4  # tokenizer downsamples 4x
        self.modal_embed_vision = nn.Parameter(torch.zeros(1, 1, d))
        self.modal_embed_wrench = nn.Parameter(torch.zeros(1, 1, d))
        self.temporal_embed_wrench = nn.Parameter(torch.zeros(1, t_tokens, d))
        for p in (self.modal_embed_vision, self.modal_embed_wrench, self.temporal_embed_wrench):
            nn.init.normal_(p, std=0.02)

        self.cross_attn = BidirectionalCrossAttention(cfg)
        self.vision_pool = AttentionPool1d(cfg, cfg.num_vision_tokens_pooled)
        self.wrench_pool = AttentionPool1d(cfg, cfg.num_wrench_tokens_pooled)

        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        n_tokens = cfg.num_vision_tokens_pooled + cfg.num_wrench_tokens_pooled + 1  # +1 state token
        self.action_head = nn.Sequential(
            nn.Linear(n_tokens * d, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, cfg.action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        state: torch.Tensor,       # (B, 24)
        wrench: torch.Tensor,      # (B, 32, 6)
        tp_rgb: torch.Tensor,      # (B, 256, 256, 3) uint8
        wrist_rgb: torch.Tensor,   # (B, 256, 256, 3) uint8
    ) -> torch.Tensor:
        tp_tok = self.tp_backbone(tp_rgb)              # (B, 64, d)
        wrist_tok = self.wrist_backbone(wrist_rgb)     # (B, 64, d)
        vision_tok = torch.cat([tp_tok, wrist_tok], dim=1) + self.modal_embed_vision  # (B, 128, d)

        wrench_tok = self.wrench_tokenizer(wrench)     # (B, 8, d)
        wrench_tok = wrench_tok + self.temporal_embed_wrench[:, : wrench_tok.shape[1]] + self.modal_embed_wrench

        vision_tok, wrench_tok = self.cross_attn(vision_tok, wrench_tok)

        vision_pooled = self.vision_pool(vision_tok)   # (B, 4, d)
        wrench_pooled = self.wrench_pool(wrench_tok)    # (B, 4, d)
        state_tok = self.state_encoder(state).unsqueeze(1)  # (B, 1, d)

        fused = torch.cat([vision_pooled, wrench_pooled, state_tok], dim=1)  # (B, 9, d)
        return self.action_head(fused.flatten(1))


if __name__ == "__main__":
    # CPU smoke test: forward + backward, shape + grad-flow sanity.
    torch.manual_seed(0)
    cfg = StudentFMTConfig(pretrained_backbone=False)  # skip weight download for smoke test
    policy = StudentFMTPolicy(cfg)

    B = 2
    state = torch.randn(B, cfg.state_dim)
    wrench = torch.randn(B, cfg.wrench_horizon, cfg.wrench_dim)
    tp_rgb = torch.randint(0, 256, (B, 256, 256, 3), dtype=torch.uint8)
    wrist_rgb = torch.randint(0, 256, (B, 256, 256, 3), dtype=torch.uint8)

    action = policy(state, wrench, tp_rgb, wrist_rgb)
    assert action.shape == (B, cfg.action_dim), action.shape
    assert bool((action >= -1).all() and (action <= 1).all()), "tanh head out of range"

    loss = action.sum()
    loss.backward()
    n_total = sum(1 for _ in policy.parameters())
    n_grad = sum(1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"SMOKE_OK action.shape={tuple(action.shape)} params={n_params:,} "
        f"grad_flow={n_grad}/{n_total}",
        flush=True,
    )
