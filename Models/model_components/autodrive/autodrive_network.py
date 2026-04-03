import torch.nn as nn
from Models.model_components.autodrive.autodrive_backbone import AutoDriveBackbone
from Models.model_components.autodrive.autodrive_head import AutoDriveHead

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

# Fixed backbone hyperparameters: [in, p1, p2, p3, p4, p5]
_WIDTH = [3, 16, 32, 64, 128, 256]
_DEPTH = [1, 1, 1, 1, 1, 1]
_CSP = [False, True]


class AutoDrive(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoDriveBackbone(_WIDTH, _DEPTH, _CSP)
        self.head = AutoDriveHead(in_channels=_WIDTH[5])  # 256 for current config

    def forward(self, x):
        p5 = self.backbone(x)
        return self.head(p5)
        # Returns: (d_norm (B,1), curvature (B,1), flag_logits (B,2))
        # distance_m = AutoDriveHead.to_distance_meters(d_norm)
