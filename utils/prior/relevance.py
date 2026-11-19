"""Ground-truth feature relevance from TabICL's SCM prior.

``RelevanceSCMPrior`` generates exactly the same tasks as ``tabicl.prior`` (same code path and the
same random draws) but additionally records the causal graph of every task and labels each final
feature column as *relevant* or not:

    relevant(X_j)  <=>  X_j is NOT d-separated from y given all other observed features X_{-j}

This is "strong relevance" (Kohavi & John): under faithfulness these are exactly the features of the
Markov boundary of y, i.e. the features an optimal feature selector must keep. The d-separation test
uses the moral graph of the ancestral set of all observed variables (Lauritzen et al.).

How the graph is built from a generated task:
  * MLP SCM: nodes = root causes, every hidden unit of every layer; an edge i -> j exists iff the
    weight W[j, i] != 0 (block-wise / Bernoulli dropout zeroes weights). All activations in
    tabicl.prior act element-wise, so the weights are the full dependency structure.
  * Tree SCM: an edge i -> j exists iff input i is used in a split of the tree model that computes
    output j (XGBoost multi-output trees share splits across outputs).
  * X and y are hidden units picked by ``handle_outputs`` (recorded here).
  * Reg2Cls may turn a feature into a categorical one: the observed variable then is a new child of
    the original unit (conditioning on a coarsened variable does not block paths through it).
  * Reg2Cls permutes the feature columns and constant columns are deleted afterwards; the labels
    follow both operations so that they align with the final X.

Caveat: labels are *structural*. A weak edge, a saturated activation or a near-deterministic unit can
make a "relevant" feature statistically useless in a finite sample (faithfulness violations).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch import nn

from tabicl.prior._dataset import SCMPrior
from tabicl.prior._mlp_scm import MLPSCM
from tabicl.prior._reg2cls import MulticlassAssigner, Reg2Cls, outlier_removing, standard_scaling
from tabicl.prior._tree_scm import TreeLayer, TreeSCM

# --------------------------------------------------------------------------- d-separation


def strongly_relevant(adjacency: np.ndarray, x_nodes: np.ndarray, y_node: int) -> np.ndarray:
    """For every observed node in ``x_nodes``: is it d-connected to ``y_node`` given the others?

    adjacency[i, j] = True  <=>  edge i -> j.
    """
    n = adjacency.shape[0]
    x_nodes = np.asarray(x_nodes)
    if len(x_nodes) == 0:
        return np.zeros(0, dtype=bool)

    # Ancestral set of all observed variables (X and y); identical for every test X_j vs y | X_{-j}
    in_set = np.zeros(n, dtype=bool)
    in_set[x_nodes] = True
    in_set[y_node] = True
    frontier = in_set.copy()
    while frontier.any():
        parents = adjacency[:, frontier].any(axis=1) & ~in_set
        in_set |= parents
        frontier = parents

    nodes = np.flatnonzero(in_set)
    position = -np.ones(n, dtype=np.int64)
    position[nodes] = np.arange(len(nodes))
    sub = adjacency[np.ix_(nodes, nodes)].astype(np.float32)

    # Moral graph: drop directions and marry parents of a common child
    # (errstate: numpy 2.1 + macOS Accelerate emits spurious "divide by zero in matmul" warnings)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        moral = (sub + sub.T + sub @ sub.T) > 0
    np.fill_diagonal(moral, False)

    # X_j - y connected in moral graph minus X_{-j}  <=>  X_j is adjacent to y's component in moral minus X
    is_x = np.zeros(len(nodes), dtype=bool)
    is_x[position[x_nodes]] = True
    rest = np.flatnonzero(~is_x)
    _, labels = connected_components(moral[np.ix_(rest, rest)], directed=False)
    y_component = rest[labels == labels[np.searchsorted(rest, position[y_node])]]
    return moral[np.ix_(position[x_nodes], y_component)].any(axis=1)


# ------------------------------------------------------------------ recording generators


class _RecordOutputsMixin:
    """Copy of ``handle_outputs`` of MLPSCM/TreeSCM (same random draws) that records the chosen units."""

    def handle_outputs(self, causes, outputs):
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if self.in_clique:
                start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
                random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
            else:
                random_perm = torch.randperm(outputs_flat.shape[-1] - 1, device=self.device)
            indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
            if self.y_is_effect:
                indices_y = list(range(-self.num_outputs, 0))
            else:
                indices_y = random_perm[: self.num_outputs]
            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]

            # Node ids: block 0 = causes, block 1 = first layer (latent), outputs_flat starts at block 2
            offset = self.num_causes + self.hidden_dim
            total = outputs_flat.shape[-1]
            self.x_nodes = (offset + torch.as_tensor(indices_X).cpu().numpy() % total).astype(np.int64)
            self.y_node = int(offset + (torch.as_tensor(indices_y).cpu().numpy() % total)[0])
        else:
            X = causes
            y = outputs[-1]
            self.x_nodes = np.arange(self.num_causes, dtype=np.int64)
            self.y_node = self._num_nodes() - self.num_outputs  # first unit of the output layer
        return X, y

    def _block_sizes(self) -> list[int]:
        sizes = [self.num_causes]
        for layer in self.layers:
            sizes.append(self._out_dim(layer))
        return sizes

    def _num_nodes(self) -> int:
        return int(sum(self._block_sizes()))

    def dependency_graph(self) -> np.ndarray:
        """adjacency[i, j] = True <=> unit i is an input of unit j."""
        sizes = self._block_sizes()
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        adjacency = np.zeros((offsets[-1], offsets[-1]), dtype=bool)
        for b, layer in enumerate(self.layers):
            dep = self._layer_dependencies(layer, sizes[b])  # (in_dim, out_dim) bool
            adjacency[offsets[b] : offsets[b + 1], offsets[b + 1] : offsets[b + 2]] = dep
        return adjacency


class RecordingMLPSCM(_RecordOutputsMixin, MLPSCM):
    @staticmethod
    def _linear(layer) -> nn.Linear:
        if isinstance(layer, nn.Linear):
            return layer
        return next(m for m in layer if isinstance(m, nn.Linear))

    def _out_dim(self, layer) -> int:
        return self._linear(layer).out_features

    def _layer_dependencies(self, layer, in_dim) -> np.ndarray:
        return (self._linear(layer).weight.detach().cpu().numpy() != 0).T


class RecordingTreeSCM(_RecordOutputsMixin, TreeSCM):
    @staticmethod
    def _tree(layer) -> TreeLayer:
        if isinstance(layer, TreeLayer):
            return layer
        return next(m for m in layer if isinstance(m, TreeLayer))

    def _out_dim(self, layer) -> int:
        return self._tree(layer).out_dim

    def _layer_dependencies(self, layer, in_dim) -> np.ndarray:
        tree = self._tree(layer)
        model = tree.model
        dep = np.zeros((in_dim, tree.out_dim), dtype=bool)
        if hasattr(model, "get_booster"):  # XGBoost multi-output trees: splits are shared by all outputs
            used = [int(name[1:]) for name in model.get_booster().get_score(importance_type="weight")]
            dep[used, :] = True
        else:  # sklearn MultiOutputRegressor: one (ensemble of) tree(s) per output
            for j, est in enumerate(model.estimators_):
                trees = getattr(est, "estimators_", [est])
                for t in np.ravel(trees):
                    feats = t.tree_.feature
                    dep[feats[feats >= 0], j] = True
        return dep


class RecordingReg2Cls(Reg2Cls):
    """Copy of Reg2Cls' feature processing (same random draws) that records categorical columns and the permutation."""

    def _num2cat(self, X):
        self.cat_columns = []
        if random.random() < self.hp.get("cat_prob", 0.2):
            col_prob = random.random()
            max_categories = self.hp.get("max_categories", 10)
            for col in range(X.shape[1]):
                if random.random() < col_prob:
                    num_cats = min(max(round(random.gammavariate(1, 10)), 2), max_categories)
                    assigner = MulticlassAssigner(num_cats, mode="rank", ordered_prob=0.3)
                    X[:, col] = assigner(X[:, col]).float()
                    self.cat_columns.append(col)
        return X

    def _process_features(self, X):
        num_features = X.shape[1]
        max_features = self.hp["max_features"]
        X = outlier_removing(X, threshold=4)
        X = standard_scaling(X)
        if self.hp.get("permute_features", True):
            perm = torch.randperm(num_features, device=X.device)
            X = X[:, perm]
        else:
            perm = torch.arange(num_features)
        self.feature_perm = perm.cpu().numpy()
        if self.hp.get("scale_by_max_features", False):
            X = X / (num_features / max_features)
        if num_features < max_features:
            X = torch.nn.functional.pad(X, (0, max_features - num_features), mode="constant", value=0.0)
        return X


class RelevanceSCMPrior(SCMPrior):
    """``tabicl.prior`` SCM prior that also returns ground-truth relevance labels per feature.

    Use ``get_batch_with_relevance()``; generation runs sequentially (n_jobs=1) so that the
    recorded labels can be matched to the generated tasks.
    """

    def __init__(self, **kwargs):
        kwargs["n_jobs"] = 1
        super().__init__(**kwargs)
        self._relevance_log: list[np.ndarray] = []

    @torch.no_grad()
    def generate_dataset(self, params):
        # Mirrors SCMPrior.generate_dataset (tabicl 2.2.0) with recording classes.
        from tabicl.prior._dataset import should_filter

        if params["prior_type"] == "mlp_scm":
            prior_cls = RecordingMLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = RecordingTreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        for idx in range(100):
            scm = prior_cls(**params)
            X, y = scm()
            reg2cls = RecordingReg2Cls(params)
            X, y = reg2cls(X, y)
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            keep = np.array([len(torch.unique(X[0, :, j])) > 1 for j in range(int(d[0]))], dtype=bool)
            X, d = self.delete_unique_features(X, d)
            if not (d > 0).all():
                continue
            if not (self.regression or self.cls_sanity_check(X, y, params["train_size"])):
                continue
            if idx <= 5 and should_filter(X[0], y[0], self.config, is_classif=not self.regression):
                continue
            break

        self._relevance_log.append(self._labels(scm, reg2cls, keep))
        return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @staticmethod
    def _labels(scm, reg2cls, keep: np.ndarray) -> np.ndarray:
        """Relevance per final column (after permutation and deletion of constant columns)."""
        adjacency = scm.dependency_graph()
        x_nodes = scm.x_nodes.copy()  # in SCM column order

        # Categorical features: the observed variable is a coarsened child of the unit
        if reg2cls.cat_columns:
            n = adjacency.shape[0]
            extra = len(reg2cls.cat_columns)
            grown = np.zeros((n + extra, n + extra), dtype=bool)
            grown[:n, :n] = adjacency
            for k, col in enumerate(reg2cls.cat_columns):
                grown[x_nodes[col], n + k] = True
                x_nodes[col] = n + k
            adjacency = grown

        observed = x_nodes[reg2cls.feature_perm][keep]  # final column order
        return strongly_relevant(adjacency, observed, scm.y_node)

    def get_batch_with_relevance(self, batch_size: int | None = None):
        """Like ``get_batch`` but also returns a list with one bool array (length d_i) per task."""
        self._relevance_log = []
        X, y, d, seq_lens, train_sizes = self.get_batch(batch_size)
        relevance = self._relevance_log
        assert len(relevance) == len(d) and all(len(r) == int(di) for r, di in zip(relevance, d))
        return X, y, d, seq_lens, train_sizes, relevance
