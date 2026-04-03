import torch
import torch.nn as nn


class AutoDriveHead(nn.Module):
    """
    Regression + classification head for AutoDrive.

    Inputs
    ------
    p5 : (B, C, H, W)  — fused P5 from AutoDriveBackbone (CTX → SPPF → C2PSA).

    Compression path
    ----------------
    AdaptiveAvgPool2d((2, 2))  → (B, C, 2, 2)
    flatten                    → (B, C*4)  = (B, 1024) for C=256

    Shared trunk
    ------------
    FC1  : Linear(1024, 768) + ReLU
    FC2  : Linear(768,  512) + ReLU

    Task branches
    -------------
    distance_head  : Linear(512, 1) + Sigmoid
                     → d ∈ (0, 1);  distance_m = 200 * (1 - d)
                     (d ≈ 0  →  ~200 m away,  d ≈ 1  →  right ahead)

    curvature_head : Linear(512, 1)  — raw regression (1/m)
                     positive = left turn, negative = right turn (or vice versa
                     depending on sign convention in labels)

    flag_head      : Linear(512, 2)  — raw logits
                     class 0 = no CIPO,  class 1 = CIPO present
                     use CrossEntropyLoss during training;
                     torch.argmax(flag_logits, dim=-1) at inference
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        # (B, C, H, W) → (B, C, 2, 2) → flatten → (B, C*4) = 1024 for C=256
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flat_dim = in_channels * 2 * 2

        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        # Distance: sigmoid so the raw output is always in (0, 1)
        self.distance_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Curvature: no activation — the network learns the sign and magnitude freely
        self.curvature_head = nn.Linear(512, 1)

        # CIPO presence flag: 2-class logits (no-CIPO | CIPO)
        self.flag_head = nn.Linear(512, 2)

    def forward(self, p5: torch.Tensor):
        x = self.pool(p5)   # (B, C, 2, 2)
        x = x.flatten(1)  # (B, 1024)

        x = self.fc1(x)                             # (B, 768)
        x = self.fc2(x)                             # (B, 512)

        d_norm      = self.distance_head(x)         # (B, 1)  ∈ (0,1)
        curvature   = self.curvature_head(x)        # (B, 1)  unbounded
        flag_logits = self.flag_head(x)             # (B, 2)  raw logits

        return d_norm, curvature, flag_logits

    @staticmethod
    def to_distance_meters(d_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised sigmoid output → metres.  distance_m = 200 * (1 - d)."""
        return 200.0 * (1.0 - d_norm)
