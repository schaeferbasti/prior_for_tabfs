"""TabFS v2: a pretrained TabICLv2 backbone with a CLS-token feature-importance readout.

TabICLv2 (the parent architecture of NVIDIA's Kumo Tabular) processes a table in three stages:

1. column-wise embedding  (induced self-attention over the rows of each column, target-aware)
2. row-wise interaction    (a transformer over the feature tokens of each row, with 4 learnable CLS
                            tokens prepended; the 4 CLS outputs are concatenated into the row embedding)
3. dataset-wise ICL        (a transformer over row embeddings that predicts the test labels)

The last row block only lets the CLS tokens query the row (q = CLS, k/v = CLS + feature tokens).
TabFS reads out how strongly ONE of these CLS tokens (``importance_cls``) attends to every feature:

    logit[b, t, h, g] = <q_cls, k_g> / sqrt(head_dim)       (pre-softmax, last row block, head h, token g)

Feature tokens in TabICLv2 are circular feature groups (token g holds features g+1, g+2, g+4 mod H),
so token logits are mapped back to features by averaging over the tokens that contain each feature.
The rows are then collapsed by a mean and the heads are combined by a learned weighted sum:

    importance[b, f] = sum_h w_h * mean_t feature_logit[b, t, h, f] + bias

Pre-softmax logits are used (not attention weights) so that the readout can be negative.

The same score is trained with one of three objectives (``objective``, see utils/tfm/train.py):
  * "relevance": score = logit of P(feature is relevant), supervised by generator labels (BCE)
  * "gating":    score = log-alpha of a hard-concrete gate (Louizos et al., 2018); no feature labels.
                 Gated-out features are replaced by noise and TabICL must still predict y (CE + L0 penalty)
  * "loco":      score = LOCO importance (regression, SmoothL1)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from tabicl._model.tabicl import TabICL

HF_REPO = "jingang/TabICL"
DEFAULT_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
OBJECTIVES = ("relevance", "gating", "loco", "ftest")  # "ftest" = learnability check, see train.py

# Hard-concrete gate (Louizos et al., 2018): stretched, clamped binary concrete distribution
HC_BETA, HC_GAMMA, HC_ZETA = 2 / 3, -0.1, 1.1


def hard_concrete_sample(log_alpha: Tensor) -> Tensor:
    u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / HC_BETA)
    return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0, 1)


def hard_concrete_deterministic(log_alpha: Tensor) -> Tensor:
    return (torch.sigmoid(log_alpha) * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0, 1)


def hard_concrete_open_probability(log_alpha: Tensor) -> Tensor:
    """P(gate != 0): the differentiable L0 penalty."""
    return torch.sigmoid(log_alpha - HC_BETA * math.log(-HC_GAMMA / HC_ZETA))


def get_default_device() -> str:
    # MPS is left out on purpose: TabICL's attention code is only tested on CPU/CUDA.
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pretrained_tabicl(checkpoint_version: str = DEFAULT_CHECKPOINT) -> tuple[TabICL, dict]:
    """Download (or read from the HF cache) a pretrained TabICL classifier checkpoint."""
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=checkpoint_version, local_files_only=True)
    except Exception:
        path = hf_hub_download(repo_id=HF_REPO, filename=checkpoint_version)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["config"]


class TabICLFS(nn.Module):
    """TabICLv2 + feature-importance readout from the attention logits of one CLS token.

    Parameters
    ----------
    backbone : TabICL
        A (pretrained) TabICL classification model.
    importance_cls : int, default=0
        Index of the CLS token (0..num_cls-1) whose attention over the features is the importance.
    objective : {"relevance", "gating", "loco"}
        How the score is trained and therefore how it is interpreted (see module docstring).
    """

    def __init__(self, backbone: TabICL, importance_cls: int = 0, objective: str = "relevance"):
        super().__init__()
        if backbone.max_classes == 0:
            raise ValueError("TabICLFS expects a TabICL classifier backbone (max_classes > 0).")
        row = backbone.row_interactor
        if not 0 <= importance_cls < row.num_cls:
            raise ValueError(f"importance_cls must be in [0, {row.num_cls - 1}], got {importance_cls}.")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}.")

        self.backbone = backbone
        self.importance_cls = importance_cls
        self.objective = objective
        num_heads = row.tf_row.blocks[-1].attn.num_heads

        # Readout parameters: per-head weights and a bias on the row-averaged CLS->feature logits.
        self.head_weights = nn.Parameter(torch.full((num_heads,), 1.0 / num_heads))
        self.bias = nn.Parameter(torch.zeros(()))
        # Targets are divided by this during training (set by train()), outputs are multiplied back
        # by feature_importance(). Stored as a buffer so it is saved with the checkpoint.
        self.register_buffer("target_scale", torch.tensor(1.0))

    def readout_parameters(self) -> list[nn.Parameter]:
        return [self.head_weights, self.bias]

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    # ------------------------------------------------------------------ forward

    def forward(self, X: Tensor, y_train: Tensor, return_predictions: bool = True) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        X : Tensor (B, T, H)
            All rows of the table(s); the first ``train_size`` rows are the labelled context.
            Features should be standardized (the prior is), see ``prediction/`` for real data.
        y_train : Tensor (B, train_size)
            Class labels 0..K-1 of the context rows (as float or long).
        return_predictions : bool
            If True, also run the ICL transformer and return logits for rows ``train_size:``.

        Returns
        -------
        importance : Tensor (B, H)
            Feature importance in normalized units (multiply by ``target_scale`` for LOCO units).
        logits : Tensor (B, T - train_size, max_classes) or None
        """
        backbone = self.backbone
        y_train = y_train.float()
        H = X.shape[-1]

        # Stage 1: column-wise embedding -> (B, T, G + C, E); first C slots are reserved for CLS
        embeddings = backbone.col_embedder._train_forward(X, y_train, d=None, embed_with_test=False)

        # Stage 2: row-wise interaction, also returning CLS->token attention logits (B, T, nh, G)
        representations, token_logits = self._row_interaction(embeddings)

        # Readout: tokens -> features, collapse rows, combine heads
        feature_logits = self._tokens_to_features(token_logits, H)  # (B, T, nh, H)
        per_head = feature_logits.mean(dim=1)  # (B, nh, H)
        importance = torch.einsum("bnh,n->bh", per_head, self.head_weights) + self.bias

        if not return_predictions:
            return importance, None

        # Stage 3: dataset-wise in-context learning (training path, no inference manager)
        train_size = y_train.shape[1]
        logits = backbone.icl_predictor._icl_predictions(representations, y_train)[:, train_size:]
        return importance, logits

    def forward_gated(self, X: Tensor, y_train: Tensor, sample: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """Self-supervised feature selection ("gating" objective).

        Pass 1 reads the CLS scores as gate log-alphas. Pass 2 replaces every feature by
        ``z * x + (1 - z) * noise`` (noise ~ N(0, 1), the scale of the standardized prior) and lets
        TabICL predict the test rows. A closed gate removes the feature's information; a soft
        multiplicative gate alone would not, because the column embedding is scale invariant.

        Returns (log_alpha (B, H), gates z (B, H), logits (B, T - train_size, max_classes)).
        """
        log_alpha, _ = self(X, y_train, return_predictions=False)
        z = hard_concrete_sample(log_alpha) if sample else hard_concrete_deterministic(log_alpha)
        gate = z.unsqueeze(1)  # (B, 1, H), same gate for every row
        X_gated = gate * X + (1 - gate) * torch.randn_like(X)
        _, logits = self(X_gated, y_train, return_predictions=True)
        return log_alpha, z, logits

    def scores_to_selection(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpret raw scores according to the objective: (probability-like value, selected mask)."""
        scores_t = torch.as_tensor(scores)
        if self.objective == "relevance":
            value = torch.sigmoid(scores_t)  # P(relevant)
            return value.numpy(), (value > 0.5).numpy()
        if self.objective == "gating":
            value = hard_concrete_deterministic(scores_t)  # gate value in [0, 1]
            return value.numpy(), (value > 0).numpy()
        # loco: LOCO units, keep features whose removal hurts; ftest: per-task z-score, above task average
        return scores, scores > 0

    def _row_interaction(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Re-implementation of ``RowInteraction._train_forward`` that also returns CLS attention logits."""
        row = self.backbone.row_interactor
        B, T, _, E = embeddings.shape
        C = row.num_cls
        rope = row.tf_row.rope

        cls_tokens = row.cls_tokens.expand(B, T, C, E)
        h = torch.cat([cls_tokens, embeddings[:, :, C:]], dim=2)  # (B, T, C + G, E)

        for block in row.tf_row.blocks[:-1]:
            h = block(h, rope=rope)

        last_block = row.tf_row.blocks[-1]
        token_logits = self._cls_attention_logits(last_block, h, rope)  # (B, T, nh, G)

        cls_outputs = last_block(q=h[..., :C, :], k=h, v=h, rope=rope)  # (B, T, C, E)
        representations = row.out_ln(cls_outputs).flatten(-2)  # (B, T, C*E)
        return representations, token_logits

    def _cls_attention_logits(self, block: nn.Module, h: Tensor, rope) -> Tensor:
        """Pre-softmax attention logits of CLS token ``importance_cls`` over the feature tokens.

        Mirrors ``MultiheadAttentionBlock.forward`` + ``multi_head_attention_forward`` for the call
        ``block(q=h[..., :C, :], k=h, v=h, rope=rope)`` up to the point where SDPA would be applied.
        """
        C = self.backbone.row_interactor.num_cls
        attn = block.attn
        E = h.shape[-1]
        num_heads = attn.num_heads
        head_dim = E // num_heads

        q_in = h[..., :C, :]
        if block.norm_first:
            q_in, k_in = block.norm1(q_in), block.norm1(h)
        else:
            k_in = h

        W, b = attn.in_proj_weight, attn.in_proj_bias
        q = F.linear(q_in, W[:E], None if b is None else b[:E])
        k = F.linear(k_in, W[E : 2 * E], None if b is None else b[E : 2 * E])
        q = q.view(*q.shape[:-1], num_heads, head_dim).transpose(-3, -2)  # (B, T, nh, C, hd)
        k = k.view(*k.shape[:-1], num_heads, head_dim).transpose(-3, -2)  # (B, T, nh, C+G, hd)
        if rope is not None:
            # RoPE positions: CLS tokens are 0..C-1 for both q and k, exactly as in the backbone
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
        if attn.ssmax_layer is not None:  # not used in TabICLv2's row stage, kept for safety
            q_shape = q.shape
            q = attn.ssmax_layer(q.reshape(-1, *q_shape[-3:]), k.size(-2)).view(q_shape)

        q_imp = q[..., self.importance_cls : self.importance_cls + 1, :]  # (B, T, nh, 1, hd)
        k_feat = k[..., C:, :]  # (B, T, nh, G, hd)
        logits = torch.matmul(q_imp, k_feat.transpose(-1, -2)).squeeze(-2) / math.sqrt(head_dim)
        return logits  # (B, T, nh, G)

    def _tokens_to_features(self, token_logits: Tensor, num_features: int) -> Tensor:
        """Map logits over feature tokens (groups) back to the original features."""
        col = self.backbone.col_embedder
        if not col.feature_group:
            return token_logits
        mode = "same" if col.feature_group is True else col.feature_group
        if mode != "same":
            raise NotImplementedError("Only the circular 'same' feature grouping of TabICLv2 is supported.")

        # In 'same' mode, token g holds features (g + 2**i) % H for i < group_size,
        # so feature f sits in token (f - 2**i) % H. Average over those tokens.
        H = num_features
        f = torch.arange(H, device=token_logits.device)
        per_slot = [token_logits[..., (f - 2**i) % H] for i in range(col.feature_group_size)]
        return torch.stack(per_slot, dim=0).mean(dim=0)

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def feature_importance(self, X: np.ndarray, y: np.ndarray, device: str | None = None) -> np.ndarray:
        """One forward pass over a whole (preprocessed) table: all rows are labelled context.

        Returns one score per feature (higher = more important): a relevance logit, a gate
        log-alpha or LOCO units, depending on ``objective`` (see ``scores_to_selection``).
        """
        device = device or get_default_device()
        self.to(device).eval()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device).unsqueeze(0)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device).unsqueeze(0)
        importance, _ = self(X_t, y_t, return_predictions=False)
        return (importance[0] * self.target_scale).cpu().numpy()


# -------------------------------------------------------------------- checkpoints


def build_tabicl_fs(
    checkpoint_version: str = DEFAULT_CHECKPOINT, importance_cls: int = 0, objective: str = "relevance"
) -> TabICLFS:
    """Pretrained TabICLv2 with a fresh (untrained) importance readout."""
    backbone, _ = load_pretrained_tabicl(checkpoint_version)
    return TabICLFS(backbone, importance_cls=importance_cls, objective=objective)


def save_tabicl_fs(model: TabICLFS, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backbone = model.backbone
    config = {
        key: getattr(backbone, key)
        for key in [
            "max_classes", "num_quantiles", "embed_dim", "col_num_blocks", "col_nhead", "col_num_inds",
            "col_affine", "col_feature_group", "col_feature_group_size", "col_target_aware", "col_ssmax",
            "row_num_blocks", "row_nhead", "row_num_cls", "row_rope_base", "row_rope_interleaved",
            "icl_num_blocks", "icl_nhead", "icl_ssmax", "ff_factor", "dropout", "activation",
            "norm_first", "bias_free_ln",
        ]
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_config": config,
            "importance_cls": model.importance_cls,
            "objective": model.objective,
            "extra": extra or {},
        },
        path,
    )


def load_tabicl_fs(
    path: str | Path | None = None, importance_cls: int = 0, objective: str = "relevance"
) -> TabICLFS:
    """Load a fine-tuned TabFS checkpoint, or the plain pretrained TabICLv2 if ``path`` is None."""
    if path is None:
        return build_tabicl_fs(importance_cls=importance_cls, objective=objective)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICLFS(
        TabICL(**checkpoint["backbone_config"]),
        importance_cls=checkpoint["importance_cls"],
        objective=checkpoint.get("objective", "loco"),  # checkpoints from before objectives were LOCO
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
