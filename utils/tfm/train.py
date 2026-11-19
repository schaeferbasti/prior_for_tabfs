"""Fine-tuning of TabICLFS on the feature-selection prior.

Objectives (``model.objective``):
  relevance  BCE(score, generator relevance label), class-balanced          + lambda_pred * CE(TabICL)
  loco       SmoothL1(score, LOCO / target_scale)                             + lambda_pred * CE(TabICL)
  gating     CE(TabICL on gated features) + lambda_l0 * mean P(gate open)     (no feature labels)
  ftest      MSE(score, per-task z-score of log1p(ANOVA F on the context rows)) + lambda_pred * CE
             -> a LEARNABILITY CHECK, not a feature-selection target: the target is a deterministic
                function of exactly the data the model sees, so a working setup must be able to fit it.

The lambda_pred * CE term keeps the CLS tokens doing what they were pretrained for.
Validation always reports AUC of the scores against the generator relevance labels (when the prior
has them), so all objectives - and the untrained readout at step 0 - are compared on the same scale.
Tasks are processed one at a time (gradient accumulation), so no padding is needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

from utils.prior.prior import PriorTasks
from utils.tfm.tabicl_fs import TabICLFS, get_default_device, hard_concrete_open_probability, save_tabicl_fs


def _to_device(task: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    train_size = task["train_size"]
    X = task["X"].unsqueeze(0).to(device)
    y = task["y"].to(device)
    return X, y, train_size


def ftest_target(task: dict) -> torch.Tensor:
    """Per-task z-score of log1p(ANOVA F) computed on the context rows only (what the model sees)."""
    ts = task["train_size"]
    X, y = task["X"][:ts].numpy(), task["y"][:ts].numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.log1p(np.nan_to_num(f_classif(X, y)[0], nan=0.0, posinf=0.0))
    f = (f - f.mean()) / f.std() if f.std() > 0 else np.zeros_like(f)
    return torch.as_tensor(f, dtype=torch.float32)


def task_loss(model: TabICLFS, task: dict, device: str, lambda_pred: float, lambda_l0: float,
              pos_weight: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
    X, y, train_size = _to_device(task, device)
    y_train, y_test = y[:train_size].unsqueeze(0), y[train_size:].long()

    if model.objective == "gating":
        log_alpha, _, logits = model.forward_gated(X, y_train, sample=True)
        ce = F.cross_entropy(logits[0], y_test)
        l0 = hard_concrete_open_probability(log_alpha[0]).mean()
        return ce + lambda_l0 * l0, {"ce": ce.item(), "open": l0.item()}

    with_prediction = lambda_pred > 0
    score, logits = model(X, y_train, return_predictions=with_prediction)
    if model.objective == "relevance":
        target = task["relevance"].to(device)
        loss_fs = F.binary_cross_entropy_with_logits(score[0], target, pos_weight=pos_weight)
    elif model.objective == "ftest":
        loss_fs = F.mse_loss(score[0], ftest_target(task).to(device))
    else:  # loco
        target = task["importance"].to(device) / model.target_scale
        loss_fs = F.smooth_l1_loss(score[0], target)

    ce = F.cross_entropy(logits[0], y_test) if with_prediction else torch.zeros((), device=device)
    return loss_fs + lambda_pred * ce, {"fs": loss_fs.item(), "ce": ce.item()}


@torch.no_grad()
def evaluate(model: TabICLFS, tasks: PriorTasks, device: str, loco_constant: float | None = None) -> dict:
    model.eval()
    scores, labels, per_task_auc, selected_frac = [], [], [], []
    spearman, mae, mae_const, gated_ce, spearman_f = [], [], [], [], []
    for k in range(len(tasks)):
        task = tasks.task(k)
        X, y, train_size = _to_device(task, device)
        y_train = y[:train_size].unsqueeze(0)
        score = model(X, y_train, return_predictions=False)[0][0]
        s = score.cpu().numpy()
        selected_frac.append(model.scores_to_selection(s)[1].mean())

        if tasks.has_relevance:
            rel = task["relevance"].numpy()
            scores.append(s)
            labels.append(rel)
            if 0 < rel.mean() < 1:
                per_task_auc.append(roc_auc_score(rel, s))
        if model.objective == "loco" and tasks.has_loco:
            pred = s * model.target_scale.item()
            target = task["importance"].numpy()
            mae.append(np.abs(pred - target).mean())
            mae_const.append(np.abs(loco_constant - target).mean())
            if len(target) >= 3 and target.std() > 0 and pred.std() > 0:
                spearman.append(spearmanr(pred, target)[0])
        if model.objective == "ftest":
            target = ftest_target(task).numpy()
            if len(target) >= 3 and target.std() > 0 and s.std() > 0:
                spearman_f.append(spearmanr(s, target)[0])
        if model.objective == "gating" and train_size < len(y):
            _, _, logits = model.forward_gated(X, y_train, sample=False)
            gated_ce.append(F.cross_entropy(logits[0], y[train_size:].long()).item())
    model.train()

    out = {"selected": float(np.mean(selected_frac))}
    if scores:
        out["auc_pooled"] = float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))
        out["auc_task"] = float(np.mean(per_task_auc)) if per_task_auc else float("nan")
    if mae:
        out.update(spearman=float(np.mean(spearman)), mae=float(np.mean(mae)), mae_const=float(np.mean(mae_const)))
    if gated_ce:
        out["gated_ce"] = float(np.mean(gated_ce))
    if spearman_f:
        out["spearman_ftest"] = float(np.mean(spearman_f))
    return out


def _format(scores: dict) -> str:
    return " | ".join(f"{k} {v:.3f}" for k, v in scores.items())


def train(
    model: TabICLFS,
    train_tasks: PriorTasks,
    val_tasks: PriorTasks | None = None,
    steps: int = 500,
    tasks_per_step: int = 8,
    lr_backbone: float = 1e-5,
    lr_readout: float = 1e-2,
    lambda_pred: float = 0.1,
    lambda_l0: float = 0.1,
    balance_classes: bool = True,
    freeze_backbone: bool = False,
    eval_every: int = 25,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[TabICLFS, list[dict]]:
    """Fine-tune the readout (and optionally the backbone) with ``model.objective``."""
    device = device or get_default_device()
    torch.manual_seed(seed)
    model.to(device).train()
    objective = model.objective

    if objective == "relevance" and not train_tasks.has_relevance:
        raise ValueError("The prior has no relevance labels; rebuild it with utils.prior.prior.build_prior.")
    if objective == "loco" and not train_tasks.has_loco:
        raise ValueError("The prior has no LOCO targets; rebuild it with n_loco_repeats > 0.")
    if objective == "gating" and freeze_backbone:
        print("Note: with a frozen backbone only the readout learns which features to gate.")

    loco_constant = None
    if objective == "loco":
        model.target_scale.fill_(train_tasks.target_std())
        loco_constant = float(np.mean([train_tasks.task(k)["importance"].mean().item()
                                       for k in range(len(train_tasks))]))
    pos_weight = None
    if objective == "relevance" and balance_classes:
        rate = train_tasks.relevance_rate()
        pos_weight = torch.tensor((1 - rate) / rate, device=device)
        print(f"relevance rate in training tasks {rate:.2f} -> pos_weight {pos_weight.item():.2f}")

    for p in model.backbone_parameters():
        p.requires_grad_(not freeze_backbone)
    groups = [{"params": model.readout_parameters(), "lr": lr_readout}]
    if not freeze_backbone:
        groups.append({"params": model.backbone_parameters(), "lr": lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    if freeze_backbone:
        lambda_pred = 0.0  # the CE term cannot change anything when the backbone is frozen

    history = []
    if val_tasks is not None:
        scores = evaluate(model, val_tasks, device, loco_constant)
        history.append({"step": 0, **scores})
        print(f"step     0 | val {_format(scores)}")

    batches = train_tasks.iterate(tasks_per_step, seed=seed)
    start = time.time()
    try:
        for step in range(1, steps + 1):
            batch = next(batches)
            logs: dict[str, float] = {}
            for task in batch:
                loss, info = task_loss(model, task, device, lambda_pred, lambda_l0, pos_weight)
                (loss / len(batch)).backward()
                for k, v in info.items():
                    logs[k] = logs.get(k, 0.0) + v / len(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % eval_every == 0 or step == steps:
                line = f"step {step:5d} | {time.time() - start:6.0f}s | train {_format(logs)}"
                if val_tasks is not None:
                    scores = evaluate(model, val_tasks, device, loco_constant)
                    history.append({"step": step, **{f"train_{k}": v for k, v in logs.items()}, **scores})
                    line += f" | val {_format(scores)}"
                print(line)
    except KeyboardInterrupt:
        print("Interrupted, saving the current model.")

    model.eval()
    if checkpoint_path is not None:
        save_tabicl_fs(model, checkpoint_path, extra={"history": history, "lambda_pred": lambda_pred,
                                                       "lambda_l0": lambda_l0})
        print(f"Saved checkpoint to {checkpoint_path}")
    return model, history
