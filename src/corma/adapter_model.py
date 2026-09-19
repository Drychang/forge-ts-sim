"""CoRMA/C3 noise-regression adapter (causal Transformer).

Architecture per CoRMA's disclosed adapter spec (d=256, 4 layers, learned
readout token). Input: (B, H, 24) sequence of deployable T-A obs. Output:
(B, 3) predicted init_fixed_pos_obs_noise (in mm; convert /1000 -> m at eval
injection). Deployed at eval by overwriting T-B obs[54:57] with this pred.
"""
import torch
import torch.nn as nn


class CausalNoiseAdapter(nn.Module):
    def __init__(self, obs_dim=24, d=256, nhead=8, layers=4, ff=512, H=32,
                 out_dim=3, target_scale=1000.0):
        super().__init__()
        self.H = H
        self.target_scale = target_scale  # predict noise in mm for conditioning
        self.in_proj = nn.Linear(obs_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, H + 1, d))          # +1 readout
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                         activation="gelu", batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                  nn.Linear(128, out_dim))

    def forward(self, seq, key_padding_mask=None):
        # seq: (B, H, 24) already normalized. key_padding_mask: (B, H) True=pad.
        b = seq.shape[0]
        x = self.in_proj(seq)                                      # (B,H,d)
        r = self.readout.expand(b, -1, -1)                         # (B,1,d)
        x = torch.cat([r, x], dim=1) + self.pos[:, : x.shape[1] + 1]
        if key_padding_mask is not None:
            pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=seq.device),
                             key_padding_mask], dim=1)             # readout never padded
        else:
            pad = None
        h = self.encoder(x, src_key_padding_mask=pad)              # (B,H+1,d)
        return self.head(h[:, 0])                                  # (B,3) in mm


if __name__ == "__main__":
    # offline smoke: forward + backward on random tensors
    m = CausalNoiseAdapter()
    b, H = 8, 32
    seq = torch.randn(b, H, 24)
    mask = torch.zeros(b, H, dtype=torch.bool); mask[:, :10] = True  # some padded
    out = m(seq, mask)
    assert out.shape == (b, 3), out.shape
    loss = ((out - torch.randn(b, 3)) ** 2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"ADAPTER_SMOKE_OK out={tuple(out.shape)} params={n_params/1e6:.2f}M loss={loss.item():.3f}")
