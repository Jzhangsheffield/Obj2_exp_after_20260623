"""Process-local model/input variants; no project source files are modified."""
from __future__ import annotations
from typing import Callable

SUPPORTED_BACKBONES = (
    "resnet3d10", "resnet3d18", "dual_resnet3d10", "dual_resnet3d18",
    "temporal_attn_resnet3d10", "temporal_attn_resnet3d18",
    "tv_r3d18", "r2plus1d18", "mvit_v2_s", "swin3d_t",
)
SUPPORTED_REPRESENTATIONS = ("rgb", "absdiff", "rgb_absdiff")
SUPPORTED_TEMPORAL_MODES = ("current", "t3_lfb")

def _apply_t3(model):
    import torch.nn as nn
    model.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    for block in (model.layer3[0], model.layer4[0]):
        block.conv1.stride = (1, 2, 2)
        block.downsample[0].stride = (1, 2, 2)
        block.stride = (1, 2, 2)
    model.backbone_temporal_mode = "t3_lfb"
    return model

def _install_input_representation(loader_module, representation: str):
    if representation not in SUPPORTED_REPRESENTATIONS: raise ValueError(representation)
    cls = loader_module.PackedRGBDepthMapDataset
    current = cls._load_rgb
    if getattr(current, "_repair_representation", None) == representation: return
    original = getattr(current, "_repair_original", current)
    def convert(x):
        if representation == "rgb": return x
        delta = x.new_zeros(x.shape)
        delta[1:] = (x[1:] - x[:-1]).abs()
        return delta if representation == "absdiff" else __import__("torch").cat((x, delta), dim=1)
    def patched(self, rec):
        out = original(self, rec)
        return tuple(convert(x) for x in out) if isinstance(out, tuple) else convert(out)
    patched._repair_original = original
    patched._repair_representation = representation
    cls._load_rgb = patched

def _make_model_factory(resnet_module, temporal_mode: str, backbone_init: str, freeze_patch_embed: bool) -> Callable:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    def base(depth: int, num_classes: int):
        model = resnet_module.generate_model(depth, num_classes=num_classes)
        return _apply_t3(model) if temporal_mode == "t3_lfb" else model
    class DualStream(nn.Module):
        def __init__(self, depth, num_classes, l2_normalize_before_fc=False):
            super().__init__()
            self.appearance = base(depth, 512); self.motion = base(depth, 512)
            self.appearance.fc = nn.Identity(); self.motion.fc = nn.Identity()
            self.fusion = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
            self.fc = nn.Linear(512, num_classes); self.l2_normalize_before_fc = bool(l2_normalize_before_fc)
        def forward_features(self, x):
            if x.shape[1] != 6: raise ValueError(f"Dual stream expects six channels [RGB, abs-difference], got {tuple(x.shape)}")
            return self.fusion(torch.cat((self.appearance.forward_features(x[:, :3]), self.motion.forward_features(x[:, 3:])), dim=1))
        def forward(self, x):
            z = self.forward_features(x)
            return self.fc(F.normalize(z, dim=1) if self.l2_normalize_before_fc else z)
    class TemporalAttention(nn.Module):
        def __init__(self, depth, num_classes, l2_normalize_before_fc=False):
            super().__init__()
            self.trunk = base(depth, 512); self.trunk.fc = nn.Identity()
            self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.fc = nn.Linear(512, num_classes); self.l2_normalize_before_fc = bool(l2_normalize_before_fc)
        def forward_features(self, x):
            x = self.trunk.forward_stem(x)
            x = self.trunk.layer1(x); x = self.trunk.layer2(x); x = self.trunk.layer3(x); x = self.trunk.layer4(x)
            seq = x.mean(dim=(-1, -2)).transpose(1, 2)
            return (seq * torch.softmax(self.attention(seq), dim=1)).sum(dim=1)
        def forward(self, x):
            z = self.forward_features(x)
            return self.fc(F.normalize(z, dim=1) if self.l2_normalize_before_fc else z)
    class TorchvisionVideoWithFC(nn.Module):
        """Give torchvision video models the same forward_features/fc interface."""
        def __init__(self, name, num_classes, l2_normalize_before_fc=False):
            super().__init__()
            from torchvision.models.video import (
                MViT_V2_S_Weights, R2Plus1D_18_Weights, R3D_18_Weights,
                Swin3D_T_Weights, mvit_v2_s, r2plus1d_18, r3d_18, swin3d_t,
            )
            use_k400 = backbone_init == "kinetics400"
            if name == "tv_r3d18":
                model = r3d_18(weights=R3D_18_Weights.DEFAULT if use_k400 else None)
                feature_dim = int(model.fc.in_features); model.fc = nn.Identity()
            elif name == "r2plus1d18":
                model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if use_k400 else None)
                feature_dim = int(model.fc.in_features); model.fc = nn.Identity()
            elif name == "mvit_v2_s":
                model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT if use_k400 else None)
                linear = [m for m in model.head.modules() if isinstance(m, nn.Linear)][-1]
                feature_dim = int(linear.in_features); model.head = nn.Identity()
            elif name == "swin3d_t":
                model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT if use_k400 else None)
                feature_dim = int(model.head.in_features); model.head = nn.Identity()
            else:
                raise ValueError(name)
            self.backbone = model
            self.fc = nn.Linear(feature_dim, int(num_classes))
            self.feature_dim = feature_dim
            self.backbone_name = name
            self.backbone_init = backbone_init
            self.l2_normalize_before_fc = bool(l2_normalize_before_fc)
            if freeze_patch_embed:
                if name == "mvit_v2_s": target = self.backbone.conv_proj
                elif name == "swin3d_t": target = self.backbone.patch_embed.proj
                else: raise ValueError("freeze_patch_embed is valid only for mvit_v2_s or swin3d_t")
                for parameter in target.parameters(): parameter.requires_grad = False
                self.patch_embed_frozen = True
        def forward_features(self, x): return self.backbone(x)
        def forward(self, x):
            z = self.forward_features(x)
            return self.fc(F.normalize(z, dim=1) if self.l2_normalize_before_fc else z)
    def generate(name, num_classes, model_depth=18, l2_normalize_before_fc=False):
        n = str(name).lower(); depth = 10 if n.endswith("10") else 18
        if n in {"resnet3d10", "resnet3d18"}:
            model = base(depth, num_classes); model.l2_normalize_before_fc = bool(l2_normalize_before_fc); return model
        if n.startswith("dual_"): return DualStream(depth, num_classes, l2_normalize_before_fc)
        if n.startswith("temporal_attn_"):
            if temporal_mode != "t3_lfb": raise ValueError("Temporal-attention variants require temporal_mode=t3_lfb")
            return TemporalAttention(depth, num_classes, l2_normalize_before_fc)
        if n in {"tv_r3d18", "r2plus1d18", "mvit_v2_s", "swin3d_t"}:
            return TorchvisionVideoWithFC(n, num_classes, l2_normalize_before_fc)
        raise ValueError(f"Unsupported repair backbone: {name}")
    return generate

def configure_partial_finetune(model):
    """Unfreeze the classifier and the final semantic stage only."""
    for parameter in model.parameters(): parameter.requires_grad = False
    for parameter in model.fc.parameters(): parameter.requires_grad = True
    name = getattr(model, "backbone_name", "")
    if name in {"tv_r3d18", "r2plus1d18"}:
        targets = [model.backbone.layer4]
    elif name == "mvit_v2_s":
        targets = [model.backbone.blocks[-2], model.backbone.blocks[-1], model.backbone.norm]
    elif name == "swin3d_t":
        targets = [model.backbone.features[-2], model.backbone.features[-1], model.backbone.norm]
    elif hasattr(model, "layer4"):
        targets = [model.layer4]
    else:
        raise ValueError(f"No partial-finetune rule for {type(model).__name__}")
    for target in targets:
        for parameter in target.parameters(): parameter.requires_grad = True
    return model

def install(src_root, representation: str, temporal_mode: str, backbone_init: str = "random", freeze_patch_embed: bool = False):
    import sys
    src = str(src_root)
    if src not in sys.path: sys.path.insert(0, src)
    if temporal_mode not in SUPPORTED_TEMPORAL_MODES: raise ValueError(temporal_mode)
    if backbone_init not in {"random", "kinetics400"}: raise ValueError(backbone_init)
    import backbone.resnet as resnet_module
    import backbone.video_backbone as video_module
    import utils_.mapstype_dataloader_with_index as loader_module
    _install_input_representation(loader_module, representation)
    generate = _make_model_factory(resnet_module, temporal_mode, backbone_init, bool(freeze_patch_embed))
    video_module.SUPPORTED_BACKBONES = SUPPORTED_BACKBONES
    video_module.generate_video_model = lambda backbone_name, num_classes, model_depth=18, l2_normalize_before_fc=False: generate(backbone_name, num_classes, model_depth, l2_normalize_before_fc)
