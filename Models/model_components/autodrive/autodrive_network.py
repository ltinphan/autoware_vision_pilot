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
    """Shared backbone on previous and current frame; head fuses both P5 features."""

    def __init__(self):
        super().__init__()
        self.backbone = AutoDriveBackbone(_WIDTH, _DEPTH, _CSP)
        self.head = AutoDriveHead(in_channels=_WIDTH[5])

    def forward(self, image_prev, image_curr):
        """Returns (distance, curvature, cipo_presence): d_norm (B,1), curvature (B,1), CIPO logits (B,2)."""
        feature_prev = self.backbone(image_prev)
        feature_curr = self.backbone(image_curr)
        return self.head(feature_prev, feature_curr)
