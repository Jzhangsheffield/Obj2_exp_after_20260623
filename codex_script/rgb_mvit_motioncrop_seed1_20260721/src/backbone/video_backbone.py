"""Unified RGB video backbones for the isolated experiment package.

Only random initialization is supported on purpose.  Both returned models expose
``fc`` so the copied MoCo and classifier code can treat ResNet3D and MViT in the
same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s

from backbone import resnet


SUPPORTED_BACKBONES = ("resnet3d18", "mvit_v2_s")


class MViTV2SWithFC(nn.Module):
    """Randomly initialized torchvision MViT-V2-S with a ResNet-like ``fc``."""

    def __init__(self, num_classes: int, l2_normalize_before_fc: bool = False):
        super().__init__()
        # weights=None is deliberate: this experiment is random-init only.
        model = mvit_v2_s(weights=None)
        linear_layers = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("Unable to locate the torchvision MViT classifier input dimension")
        feature_dim = int(linear_layers[-1].in_features)
        model.head = nn.Identity()
        self.backbone = model
        self.fc = nn.Linear(feature_dim, int(num_classes))
        self.feature_dim = feature_dim
        self.l2_normalize_before_fc = bool(l2_normalize_before_fc)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"MViT expects [B,C,T,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (16, 224, 224):
            raise ValueError(
                "This MViT-V2-S experiment is locked to T=16 and H=W=224; "
                f"received {tuple(x.shape[2:])}"
            )
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.l2_normalize_before_fc:
            features = F.normalize(features, dim=1)
        return self.fc(features)


def generate_video_model(
    backbone_name: str,
    num_classes: int,
    model_depth: int = 18,
    l2_normalize_before_fc: bool = False,
) -> nn.Module:
    name = str(backbone_name).strip().lower()
    if name == "resnet3d18":
        if int(model_depth) != 18:
            raise ValueError("This experiment package locks ResNet3D to depth 18")
        return resnet.generate_model(
            18,
            num_classes=int(num_classes),
            l2_normalize_before_fc=bool(l2_normalize_before_fc),
        )
    if name == "mvit_v2_s":
        return MViTV2SWithFC(
            num_classes=int(num_classes),
            l2_normalize_before_fc=bool(l2_normalize_before_fc),
        )
    raise ValueError(f"Unsupported backbone {backbone_name!r}; choose from {SUPPORTED_BACKBONES}")

