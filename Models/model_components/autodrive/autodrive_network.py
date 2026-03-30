import torch
import torch.nn as nn
from Models.model_components.autodrive.autodrive_backbone import AutoDriveBackbone
from Models.model_components.common_layers import Conv

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

# Fixed backbone hyperparameters: [in, p1, p2, p3, p4, p5]
_WIDTH = [3, 16, 32, 64, 128, 256]
_DEPTH = [1, 1, 1, 1, 1, 1]
_CSP = [False, True]


def fuse_conv(conv, norm):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d for inference."""
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class AutoDrive(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoDriveBackbone(_WIDTH, _DEPTH, _CSP)
        # self.head = AutoDriveHead(...)  — wire when head is ready

    def forward(self, x):
        p5 = self.backbone(x)
        # return self.head(p5)
        return p5

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
