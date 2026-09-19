"""P1 architecture ablation: alternative fusion architectures for the student.

Scientific design: these reuse StudentFMT's EXACT encoders (CameraBackbone x2,
WrenchTokenizer, state MLP) and differ ONLY in how the modalities are fused.
So the comparison isolates the contribution of FMT's bidirectional
cross-attention + attention pooling, not encoder capacity or data.

  fmt     (reference) vision<->wrench bidirectional cross-attention
                      -> attention pooling (4+4 tokens) -> concat state -> MLP
  concat  (ablation)  mean-pool vision, mean-pool wrench, concat state -> MLP
                      = "no cross-modal interaction at all"
  act     (competitor) all tokens (vision + wrench + state) in ONE sequence
                      through a standard TransformerEncoder (joint
                      self-attention, the fusion used by ACT / most
                      multimodal IL transformers) -> readout token -> MLP

All expose the identical forward signature so train/eval harnesses are shared:
    policy(state (B,24), wrench (B,32,6), tp_rgb (B,256,256,3) uint8,
           wrist_rgb (B,256,256,3) uint8) -> action (B,7)
"""
import torch
import torch.nn as nn

from student_fmt import CameraBackbone, StudentFMTConfig, WrenchTokenizer


class ConcatFusionPolicy(nn.Module):
    """Late-fusion baseline: no cross-modal attention, mean pooling only."""

    def __init__(self, cfg: StudentFMTConfig = StudentFMTConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.proj_dim
        self.tp_backbone = CameraBackbone(cfg)
        self.wrist_backbone = CameraBackbone(cfg)
        self.wrench_tokenizer = WrenchTokenizer(cfg)
        self.state_encoder = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d))
        # Match the reference head: it consumes 9 tokens (4 vision + 4 wrench
        # + 1 state) flattened. Here we feed 3 pooled vectors, so widen the
        # first layer to keep head capacity comparable.
        self.action_head = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, cfg.action_dim), nn.Tanh(),
        )

    def forward(self, state, wrench, tp_rgb, wrist_rgb):
        tp_tok = self.tp_backbone(tp_rgb)
        wrist_tok = self.wrist_backbone(wrist_rgb)
        vision = torch.cat([tp_tok, wrist_tok], dim=1).mean(dim=1)      # (B,d)
        wr = self.wrench_tokenizer(wrench).mean(dim=1)                   # (B,d)
        st = self.state_encoder(state)                                   # (B,d)
        return self.action_head(torch.cat([vision, wr, st], dim=-1))


class ACTStyleFusionPolicy(nn.Module):
    """ACT-style: one token sequence, joint self-attention, readout token."""

    def __init__(self, cfg: StudentFMTConfig = StudentFMTConfig(), n_layers: int = 4):
        super().__init__()
        self.cfg = cfg
        d = cfg.proj_dim
        self.tp_backbone = CameraBackbone(cfg)
        self.wrist_backbone = CameraBackbone(cfg)
        self.wrench_tokenizer = WrenchTokenizer(cfg)
        self.state_encoder = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d))

        self.modal_vision = nn.Parameter(torch.zeros(1, 1, d))
        self.modal_wrench = nn.Parameter(torch.zeros(1, 1, d))
        self.modal_state = nn.Parameter(torch.zeros(1, 1, d))
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.modal_vision, self.modal_wrench, self.modal_state, self.readout):
            nn.init.trunc_normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.num_heads, dim_feedforward=4 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, cfg.action_dim), nn.Tanh(),
        )

    def forward(self, state, wrench, tp_rgb, wrist_rgb):
        b = state.shape[0]
        tp_tok = self.tp_backbone(tp_rgb)
        wrist_tok = self.wrist_backbone(wrist_rgb)
        vision = torch.cat([tp_tok, wrist_tok], dim=1) + self.modal_vision   # (B,128,d)
        wr = self.wrench_tokenizer(wrench) + self.modal_wrench               # (B,8,d)
        st = self.state_encoder(state).unsqueeze(1) + self.modal_state       # (B,1,d)
        ro = self.readout.expand(b, -1, -1)                                  # (B,1,d)
        seq = torch.cat([ro, vision, wr, st], dim=1)                         # (B,138,d)
        out = self.encoder(seq)
        return self.action_head(out[:, 0])


ARCHS = {"concat": ConcatFusionPolicy, "act": ACTStyleFusionPolicy}


def build_policy(arch: str, cfg: StudentFMTConfig = None):
    cfg = cfg or StudentFMTConfig()
    if arch == "fmt":
        from student_fmt import StudentFMTPolicy
        return StudentFMTPolicy(cfg)
    if arch not in ARCHS:
        raise SystemExit(f"unknown arch {arch}; choose from fmt/{'/'.join(ARCHS)}")
    return ARCHS[arch](cfg)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = StudentFMTConfig(pretrained_backbone=False)
    b = 2
    state = torch.randn(b, cfg.state_dim)
    wrench = torch.randn(b, cfg.wrench_horizon, cfg.wrench_dim)
    tp = torch.randint(0, 255, (b, 256, 256, 3), dtype=torch.uint8)
    wr = torch.randint(0, 255, (b, 256, 256, 3), dtype=torch.uint8)
    for name in ["fmt", "concat", "act"]:
        m = build_policy(name, cfg)
        a = m(state, wrench, tp, wr)
        assert a.shape == (b, cfg.action_dim), (name, a.shape)
        a.sum().backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:7s} out={tuple(a.shape)} params={n/1e6:.2f}M OK")
    print("ARCHS_SMOKE_OK")
