"""Class-conditional teacher-assigned ProtoLoss-v2 and RelLoss-v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=1e-12)


def _class_mean(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    parts = [values[labels == c].mean() for c in torch.unique(labels).tolist() if bool((labels == c).any())]
    return torch.stack(parts).mean() if parts else values.new_zeros(())


def _sinkhorn(scores: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    n, k = scores.shape
    if n < k or k <= 1:
        return torch.softmax(scores / epsilon, dim=1)
    q = torch.exp((scores / epsilon).clamp(-50, 50)).t()
    q = q / q.sum().clamp_min(1e-12)
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12) / k
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12) / n
    return (q * n).t()


@dataclass
class V2Config:
    assignment_mode: str = "teacher_balanced"
    assignment_temperature: float = 0.05
    prediction_temperature: float = 0.07
    sinkhorn_iterations: int = 3
    balance_weight: float = 0.2
    diversity_weight: float = 0.1
    diversity_cos_margin: float = 0.85
    preview_momentum: float = 0.5
    bank_momentum: float = 0.99
    rel_mode: str = "rank_direction"
    rel_topk_classes: int = 3
    rel_margin: float = 0.05
    rel_temperature: float = 0.05
    direction_weight: float = 0.25
    direction_delta: float = 0.005
    diagnostic_interval: int = 50
    diagnostic_path: Optional[str] = None


class V2Controller:
    def __init__(self, cfg: V2Config):
        self.cfg = cfg
        self.diagnostic_path = Path(cfg.diagnostic_path) if cfg.diagnostic_path else None
        self._last_bank_update_mean = 0.0
        self._last_bank_update_max = 0.0

    def _responsibilities(self, source, labels, bank, counts):
        source, bank = _norm(source), _norm(bank)
        bsz, max_proto = source.shape[0], bank.shape[1]
        resp = source.new_zeros((bsz, max_proto))
        for c in torch.unique(labels).tolist():
            c = int(c); mask = labels == c
            k = int(counts[c].item()) if 0 <= c < counts.numel() else 0
            if k <= 0:
                continue
            scores = source[mask] @ bank[c, :k].t()
            rc = _sinkhorn(scores.detach(), self.cfg.assignment_temperature, self.cfg.sinkhorn_iterations) if self.cfg.assignment_mode == "teacher_balanced" else torch.softmax(scores.detach() / self.cfg.assignment_temperature, dim=1)
            resp[mask, :k] = rc
        hard = resp.argmax(dim=1)
        hard = torch.where(resp.sum(dim=1) > 0, hard, torch.full_like(hard, -1))
        return resp.detach(), hard

    def _preview_bank(self, bank, q, labels, resp, counts):
        old, preview, qn = _norm(bank).detach(), _norm(bank).detach().clone(), _norm(q)
        for c in torch.unique(labels).tolist():
            c = int(c); mask = labels == c; k = int(counts[c].item())
            for j in range(k):
                w = resp[mask, j]
                if float(w.sum().detach()) <= 1e-8:
                    continue
                mean = _norm(((w.unsqueeze(1) * qn[mask]).sum(dim=0, keepdim=True) / w.sum().clamp_min(1e-8))).squeeze(0)
                preview[c, j] = _norm((self.cfg.preview_momentum * old[c, j] + (1 - self.cfg.preview_momentum) * mean).unsqueeze(0)).squeeze(0)
        return preview

    def _proto_loss(self, q, labels, bank, counts, resp, preview):
        qn, bn = _norm(q), _norm(bank)
        cnum, max_proto, dim = bn.shape
        logits = qn @ bn.reshape(cnum * max_proto, dim).t() / self.cfg.prediction_temperature
        active = (torch.arange(max_proto, device=q.device).unsqueeze(0) < counts.unsqueeze(1)).reshape(-1)
        # A finite sentinel avoids the undefined 0 * (-inf) term when the
        # soft target has zero mass on padded/inactive prototype slots.
        logits = logits.masked_fill(~active.unsqueeze(0), -1.0e9)
        target = torch.zeros_like(logits)
        for i in range(q.shape[0]):
            c = int(labels[i].item())
            if 0 <= c < cnum:
                target[i, c * max_proto:(c + 1) * max_proto] = resp[i]
        valid = target.sum(dim=1) > 0
        assign = -(target[valid] * F.log_softmax(logits[valid], dim=1)).sum(dim=1).mean() if bool(valid.any()) else q.new_zeros(())
        balance_terms = []
        for c in torch.unique(labels).tolist():
            c = int(c); mask = labels == c; k = int(counts[c].item())
            if k > 1 and bool(mask.any()):
                mean_r = resp[mask, :k].mean(dim=0).clamp_min(1e-8)
                balance_terms.append((mean_r * (mean_r.log() + math.log(k))).sum())
        balance = torch.stack(balance_terms).mean() if balance_terms else q.new_zeros(())
        div_terms, pn = [], _norm(preview)
        for c in range(cnum):
            k = int(counts[c].item())
            if k > 1:
                sim = pn[c, :k] @ pn[c, :k].t()
                tri = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
                div_terms.append(F.relu(sim[tri] - self.cfg.diversity_cos_margin).mean())
        diversity = torch.stack(div_terms).mean() if div_terms else q.new_zeros(())
        total = assign + self.cfg.balance_weight * balance + self.cfg.diversity_weight * diversity
        return total, {"proto_assign": assign, "proto_balance": balance, "proto_diversity": diversity}

    def _relation_loss(self, q, labels, bank, counts, resp, preview):
        qn, old, new = _norm(q), _norm(bank).detach(), _norm(preview)
        cnum = old.shape[0]

        def scores(which):
            pos = qn.new_zeros(qn.shape[0]); cls = qn.new_full((qn.shape[0], cnum), float("-inf"))
            for c in range(cnum):
                k = int(counts[c].item())
                if k <= 0: continue
                s = qn @ which[c, :k].t(); cls[:, c] = s.max(dim=1).values
                own = labels == c
                if bool(own.any()): pos[own] = (resp[own, :k] * s[own]).sum(dim=1)
            cls.scatter_(1, labels.view(-1, 1), float("-inf"))
            return pos, cls

        pos_old, neg_old_all = scores(old)
        topk = min(self.cfg.rel_topk_classes, max(cnum - 1, 1))
        _, hard_cls = torch.topk(neg_old_all.detach(), k=topk, dim=1)
        neg_old = neg_old_all.gather(1, hard_cls)
        rank_terms = F.softplus((neg_old - pos_old.unsqueeze(1) + self.cfg.rel_margin) / self.cfg.rel_temperature) * self.cfg.rel_temperature
        rank = _class_mean(rank_terms.mean(dim=1), labels)
        direction = q.new_zeros(())
        if self.cfg.rel_mode == "rank_direction":
            pos_new, neg_new_all = scores(new); neg_new = neg_new_all.gather(1, hard_cls)
            old_gap = pos_old.detach().unsqueeze(1) - neg_old.detach(); new_gap = pos_new.unsqueeze(1) - neg_new
            gate = (old_gap < self.cfg.rel_margin).float()
            vals = F.softplus((old_gap + self.cfg.direction_delta - new_gap) / self.cfg.rel_temperature) * self.cfg.rel_temperature
            direction = _class_mean((gate * vals).sum(dim=1) / gate.sum(dim=1).clamp_min(1.0), labels)
        total = rank + self.cfg.direction_weight * direction
        return total, {"rel_rank": rank, "rel_direction": direction, "hard_negative_similarity": neg_old.mean(), "margin_violation": (neg_old - pos_old.unsqueeze(1) + self.cfg.rel_margin > 0).float().mean()}

    def compute(self, q, teacher, labels, prototype_bank, class_num_prototypes, use_proto_loss, use_rel_loss, epoch, step):
        source = q if self.cfg.assignment_mode == "same_view_soft" else teacher
        resp, hard = self._responsibilities(source, labels, prototype_bank, class_num_prototypes)
        preview = self._preview_bank(prototype_bank, q, labels, resp, class_num_prototypes)
        zero = q.new_zeros(())
        proto, ps = self._proto_loss(q, labels, prototype_bank, class_num_prototypes, resp, preview) if use_proto_loss else (zero, {"proto_assign": zero, "proto_balance": zero, "proto_diversity": zero})
        rel, rs = self._relation_loss(q, labels, prototype_bank, class_num_prototypes, resp, preview) if use_rel_loss else (zero, {"rel_rank": zero, "rel_direction": zero, "hard_negative_similarity": zero, "margin_violation": zero})
        out = {"loss_proto": proto, "loss_rel": rel, "hard_ids": hard, "responsibilities": resp, **ps, **rs}
        if self.diagnostic_path and (step == 1 or step % self.cfg.diagnostic_interval == 0) and (not dist.is_initialized() or dist.get_rank() == 0):
            self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            valid = resp.sum(dim=1) > 0
            entropy = -(resp[valid].clamp_min(1e-8) * resp[valid].clamp_min(1e-8).log()).sum(dim=1).mean() if bool(valid.any()) else zero
            cnum, max_proto = prototype_bank.shape[:2]
            hard_counts = torch.zeros((cnum, max_proto), device=q.device)
            soft_mass = torch.zeros_like(hard_counts)
            for c in torch.unique(labels).tolist():
                c = int(c); mask = labels == c; k = int(class_num_prototypes[c].item())
                if k > 0 and bool(mask.any()):
                    soft_mass[c, :k] = resp[mask, :k].sum(dim=0)
                    ids = hard[mask]; ids = ids[ids >= 0]
                    if ids.numel(): hard_counts[c, :k] = torch.bincount(ids, minlength=k)[:k]
            active = torch.arange(max_proto, device=q.device).unsqueeze(0) < class_num_prototypes.unsqueeze(1)
            present = torch.zeros(cnum, dtype=torch.bool, device=q.device)
            present[torch.unique(labels).long()] = True
            observed_active = active & present.unsqueeze(1)
            active_mass = soft_mass[observed_active]
            active_hard = hard_counts[observed_active]
            cosines = []
            bankn = _norm(prototype_bank)
            for c in range(cnum):
                k = int(class_num_prototypes[c].item())
                if k > 1:
                    sim = bankn[c, :k] @ bankn[c, :k].t()
                    cosines.append(sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)])
            cosine_values = torch.cat(cosines) if cosines else zero.reshape(1)
            payload = {
                "epoch": int(epoch + 1), "step": int(step),
                "assignment_entropy": float(entropy.detach()),
                "assignment_soft_mass_min": float(active_mass.min()) if active_mass.numel() else 0.0,
                "assignment_soft_mass_max": float(active_mass.max()) if active_mass.numel() else 0.0,
                "dead_prototypes_in_batch": int((active_hard == 0).sum().item()),
                "same_class_proto_cos_mean": float(cosine_values.mean()),
                "same_class_proto_cos_max": float(cosine_values.max()),
                "bank_update_mean": self._last_bank_update_mean,
                "bank_update_max": self._last_bank_update_max,
                "assignment_hard_counts": hard_counts.detach().cpu().tolist(),
                "assignment_soft_mass": soft_mass.detach().cpu().tolist(),
                **{k: float(v.detach()) for k, v in {**ps, **rs}.items()},
            }
            with self.diagnostic_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload) + "\n")
            if (epoch + 1) % 10 == 0:
                snapshot = self.diagnostic_path.parent / f"v2_prototype_diagnostic_epoch_{epoch + 1:04d}.json"
                snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @torch.no_grad()
    def update_bank_(self, prototype_bank, teacher, labels, responsibilities, class_num_prototypes):
        teacher = _norm(teacher.detach()); cnum, max_proto, dim = prototype_bank.shape
        sums = torch.zeros((cnum, max_proto, dim), device=teacher.device); weights = torch.zeros((cnum, max_proto, 1), device=teacher.device)
        for c in torch.unique(labels).tolist():
            c = int(c); mask = labels == c; k = int(class_num_prototypes[c].item())
            if k > 0:
                sums[c, :k] = responsibilities[mask, :k].t() @ teacher[mask]
                weights[c, :k, 0] = responsibilities[mask, :k].sum(dim=0)
        if dist.is_available() and dist.is_initialized(): dist.all_reduce(sums); dist.all_reduce(weights)
        active = weights.squeeze(-1) > 1e-8
        if bool(active.any()):
            mean = _norm(sums[active] / weights[active].clamp_min(1e-8)); old = _norm(prototype_bank[active])
            updated = _norm(self.cfg.bank_momentum * old + (1 - self.cfg.bank_momentum) * mean)
            delta = (updated - old).norm(dim=1)
            self._last_bank_update_mean = float(delta.mean())
            self._last_bank_update_max = float(delta.max())
            prototype_bank[active] = updated.to(prototype_bank.dtype)
