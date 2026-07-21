#!/usr/bin/env python3
"""Temporal-stride variants for the project's existing 3D ResNet.

The T3 variant keeps the parameter shapes unchanged.  It changes only pooling
and stride metadata, so checkpoints remain structurally compatible while the
model retains eight temporal positions for a 16-frame input before avgpool.
"""

from __future__ import annotations


SUPPORTED_MODES = ("current", "t3_lfb")


def _set_spatial_only_stride(block) -> None:
    """Set the first convolution and shortcut of a transition block to (1,2,2)."""
    if hasattr(block, "conv1") and tuple(block.conv1.stride) != (1, 1, 1):
        block.conv1.stride = (1, 2, 2)
    elif hasattr(block, "conv2") and tuple(block.conv2.stride) != (1, 1, 1):
        block.conv2.stride = (1, 2, 2)
    else:
        raise RuntimeError("Could not locate the strided convolution in a ResNet transition block")
    if block.downsample is None or not hasattr(block.downsample[0], "stride"):
        raise RuntimeError("Expected a convolutional downsample shortcut")
    block.downsample[0].stride = (1, 2, 2)
    block.stride = (1, 2, 2)


def apply_t3_lfb_stride(model):
    """Apply the T3 schedule: 16 -> 16 -> 8 -> 8 -> 8 before avgpool.

    Stem pool performs spatial-only pooling. Layer2 retains the single temporal
    stride of two. Layer3 and layer4 downsample only spatial dimensions.
    """
    import torch.nn as nn

    model.maxpool = nn.MaxPool3d(
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )
    _set_spatial_only_stride(model.layer3[0])
    _set_spatial_only_stride(model.layer4[0])
    model.backbone_temporal_mode = "t3_lfb"
    model.temporal_downsample_factor = 2
    return model


def install_generate_model_mode(resnet_module, mode: str) -> None:
    """Wrap ``backbone.resnet.generate_model`` for this process only."""
    if mode not in SUPPORTED_MODES:
        raise ValueError("Unsupported backbone temporal mode: %s" % mode)
    if mode == "current":
        return
    original = resnet_module.generate_model
    if getattr(original, "_required_temporal_mode", None) == mode:
        return

    def generate_model(model_depth, **kwargs):
        model = original(model_depth, **kwargs)
        if int(model_depth) != 18:
            raise ValueError("The packaged T3 variant is validated only for ResNet3D-18")
        return apply_t3_lfb_stride(model)

    generate_model._required_temporal_mode = mode
    resnet_module.generate_model = generate_model

