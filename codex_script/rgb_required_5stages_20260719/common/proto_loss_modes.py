#!/usr/bin/env python3
"""Prototype-positive definitions used by Stage 1 and later stages."""

from __future__ import annotations

from typing import Optional

import torch


def prototype_contrastive_loss_soft_responsibility(
    q: torch.Tensor,
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    prototype_bank: torch.Tensor,
    class_num_prototypes: torch.Tensor,
    temperature: float = 0.07,
    use_prototype_temperature_scaling: bool = False,
    proto_rel_temperature_bank: Optional[torch.Tensor] = None,
    temperature_eps: float = 1e-6,
) -> torch.Tensor:
    """Soft multi-positive prototype contrastive loss.

    Same-class prototypes receive detached similarity-based responsibilities.
    This preserves multiple modes without forcing an anchor to be equally close
    to every same-class prototype. All active prototypes remain in denominator.
    """
    if q.ndim != 2 or prototype_bank.ndim != 3:
        raise ValueError("q must be [B,D] and prototype_bank must be [C,M,D]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    bsz, dim = q.shape
    nclass, max_proto, bank_dim = prototype_bank.shape
    if dim != bank_dim or labels.numel() != bsz or proto_ids.numel() != bsz:
        raise ValueError("incompatible prototype loss shapes")
    device = q.device
    labels = labels.to(device=device).long()
    proto_ids = proto_ids.to(device=device).long()
    counts = class_num_prototypes.to(device=device).long()
    valid_label = (labels >= 0) & (labels < nclass)
    safe_label = labels.clamp(0, max(nclass - 1, 0))
    valid = valid_label & (proto_ids >= 0) & (proto_ids < counts[safe_label])
    if not bool(valid.any()):
        return q.new_zeros(())
    qv = q[valid]
    yv = labels[valid]
    bank = prototype_bank.to(device=device).reshape(nclass * max_proto, dim)
    similarities = torch.matmul(qv, bank.t())
    slots = torch.arange(max_proto, device=device).unsqueeze(0)
    active2d = slots < counts.unsqueeze(1)
    active = active2d.reshape(-1)
    if use_prototype_temperature_scaling:
        if proto_rel_temperature_bank is None:
            raise ValueError("relative temperature bank is required")
        rel_tau = proto_rel_temperature_bank.to(device=device, dtype=similarities.dtype).reshape(-1)
        logits = similarities / (temperature * rel_tau.clamp_min(temperature_eps)).unsqueeze(0)
    else:
        logits = similarities / temperature
    logits = logits.masked_fill(~active.unsqueeze(0), float("-inf"))
    proto_classes = torch.arange(nclass, device=device).unsqueeze(1).expand(nclass, max_proto).reshape(-1)
    positive = active.unsqueeze(0) & proto_classes.unsqueeze(0).eq(yv.unsqueeze(1))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    responsibilities = torch.softmax(positive_logits, dim=1).detach()
    responsibilities = torch.where(positive, responsibilities, torch.zeros_like(responsibilities))
    safe_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob))
    return -(responsibilities * safe_log_prob).sum(dim=1).mean()


def get_proto_loss(mode: str):
    from loss.prorotype_contrastive_loss_mapstyle_varproto import (
        prototype_contrastive_loss_all_positive,
        prototype_contrastive_loss_single_positive,
    )
    mapping = {
        "all": prototype_contrastive_loss_all_positive,
        "single": prototype_contrastive_loss_single_positive,
        "soft": prototype_contrastive_loss_soft_responsibility,
    }
    if mode not in mapping:
        raise ValueError("Unknown prototype positive mode: %s" % mode)
    return mapping[mode]

