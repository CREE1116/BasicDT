from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._basicdt import BasicDTree, BasicDContext, update_gradients, predict_ensemble


class BasicDTClassifier(BaseEstimator, ClassifierMixin):
    """
    BasicDT: Fast histogram-based axis-aligned decision tree classifier.

    A lightweight, high-performance standalone decision tree engine using
    optimized C++ sample-parallelism and dynamic thread scaling. Strips out all
    oblique tree search logic (like random projections/mutations/Gram-Schmidt).

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage applied to each tree's leaf values.
    max_depth : int
        Maximum tree depth; leaf budget is 2^max_depth.
    reg_lambda : float
        L2 regularisation on leaf weights (Newton step denominator).
    subsample : float
        Fraction of training samples used to build each tree.
    early_stopping_rounds : int or None
        Stop if validation loss does not improve for this many rounds.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        Print per-round metrics during training.
    cat_features : list of str or int, optional
        Column names (if X is a DataFrame) or column indices treated as
        categorical.
    class_weight : str or None
        "balanced" applies a prior-corrected argmax decision rule in
        ``predict``.
    prior_alpha : float
        Strength of the prior correction used when ``class_weight == "balanced"``.
    """

    def __init__(
        self,
        n_estimators:          int   = 1000,
        learning_rate:         float = 0.03,
        max_depth:             int   = 6,
        reg_lambda:            float = 1.0,
        subsample:             float = 0.8,
        early_stopping_rounds: int | None = 50,
        random_state:          int | None = None,
        verbose:               bool  = False,
        cat_features:          list | None = None,
        class_weight:          str | None = None,
        prior_alpha:           float = 0.5,
        n_jobs:                int | None = None,
    ):
        self.n_estimators          = n_estimators
        self.learning_rate         = learning_rate
        self.max_depth             = max_depth
        self.reg_lambda            = reg_lambda
        self.subsample             = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state
        self.verbose               = verbose
        self.cat_features          = cat_features
        self.class_weight          = class_weight
        self.prior_alpha           = prior_alpha
        # OpenMP team size. Histogram building is memory-bandwidth bound, so
        # the fastest thread count is machine-specific and often below the
        # core count (~4 on Apple-silicon's shared bus; more on a many-core
        # server). None = all cores; tune per machine for best throughput.
        self.n_jobs                = n_jobs

    def _prepare_data(self, X, is_fit=False):
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, "values"):
                X = X.values
            return np.asarray(X, dtype=np.float32)

        X_prep = X.copy()

        if is_fit:
            self.feature_names_in_ = list(X.columns)
            self._cat_mappings_ = {}

        cat_cols = []
        if self.cat_features:
            for cf in self.cat_features:
                if isinstance(cf, (int, np.integer)):
                    if 0 <= cf < len(X.columns):
                        cat_cols.append(X.columns[cf])
                elif cf in X.columns:
                    cat_cols.append(cf)
        else:
            for col in X.columns:
                dtype_str = str(X[col].dtype)
                if (isinstance(X[col].dtype, pd.CategoricalDtype) or
                        dtype_str in ("string", "str", "object", "category")):
                    cat_cols.append(col)

        if is_fit and not self.cat_features and cat_cols:
            self.cat_features = cat_cols

        for col in cat_cols:
            if not isinstance(X_prep[col].dtype, pd.CategoricalDtype):
                X_prep[col] = X_prep[col].astype('category')

            if is_fit:
                self._cat_mappings_[col] = list(X_prep[col].cat.categories)
            else:
                if hasattr(self, "_cat_mappings_") and col in self._cat_mappings_:
                    X_prep[col] = pd.Categorical(
                        X_prep[col], categories=self._cat_mappings_[col]
                    )

            codes = X_prep[col].cat.codes.astype(np.float32)
            codes[codes == -1] = np.nan
            X_prep[col] = codes

        return X_prep.values.astype(np.float32)

    def fit(
        self,
        X,
        y,
        eval_set: list[tuple] | None = None,
        sample_weight=None,
    ) -> BasicDTClassifier:
        """
        Fit the classifier.
        """
        X = self._prepare_data(X, is_fit=True)
        y = np.asarray(y, dtype=np.int64)

        self.n_features_in_ = X.shape[1]
        self.classes_       = np.unique(y)

        D_num = self._resolve_D_num(X.shape[1])

        X_val, y_val = None, None
        if eval_set:
            X_val, y_val = eval_set[0]
            X_val = self._prepare_data(X_val, is_fit=False)
            y_val = np.asarray(y_val, dtype=np.int64)

        self._fit_core(X, y, X_val, y_val, D_num)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "trees_")
        P = self.predict_proba(X)
        alpha = getattr(self, "prior_alpha", 0.5)
        if (getattr(self, "class_weight", None) == "balanced"
                and getattr(self, "_prior_", None) is not None
                and alpha > 0.0):
            prior = np.asarray(self._prior_, dtype=np.float32)[None, :]
            P = P / np.power(prior, alpha)
        return P.argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "trees_")

        X = self._prepare_data(X, is_fit=False)
        X = np.ascontiguousarray(X, dtype=np.float32)
        if getattr(self, "_col_perm_", None) is not None:
            X = np.ascontiguousarray(X[:, self._col_perm_])
        N = X.shape[0]

        K = len(self.F_init_)
        F = predict_ensemble(self.trees_, X, K, self.learning_rate,
                             np.array(self.F_init_, dtype=np.float32))

        Fsh = F - F.max(axis=1, keepdims=True)
        P   = np.exp(Fsh); P /= P.sum(axis=1, keepdims=True)
        return P

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        import joblib
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> BasicDTClassifier:
        """Load a model saved with :meth:`save`."""
        import joblib
        return joblib.load(path)

    def get_n_trees(self) -> int:
        """Return the number of trees actually fitted."""
        check_is_fitted(self, "trees_")
        return len(self.trees_)

    def _resolve_cat_idx(self, D: int) -> list[int]:
        if not self.cat_features:
            return []
        cat_idx = set()
        for cf in self.cat_features:
            if isinstance(cf, (int, np.integer)):
                cat_idx.add(int(cf))
            elif self.feature_names_in_ is not None and cf in self.feature_names_in_:
                cat_idx.add(self.feature_names_in_.index(cf))
        return sorted(cat_idx)

    def _resolve_D_num(self, D: int) -> int:
        return D - len(self._resolve_cat_idx(D))

    def _fit_core(self, X, y, X_val, y_val, D_num):
        N, D = X.shape
        K    = int(y.max()) + 1
        seed = self.random_state if self.random_state is not None else 42

        if self.n_jobs is not None:
            from ._basicdt import set_num_threads
            set_num_threads(self.n_jobs)

        cat_idx = self._resolve_cat_idx(D)
        if cat_idx and cat_idx != list(range(D_num, D)):
            perm = [i for i in range(D) if i not in set(cat_idx)] + cat_idx
            self._col_perm_ = np.asarray(perm, dtype=np.intp)
        else:
            self._col_perm_ = None
        if self._col_perm_ is not None:
            X = np.ascontiguousarray(X[:, self._col_perm_])
            if X_val is not None:
                X_val = np.ascontiguousarray(X_val[:, self._col_perm_])

        cnt = np.bincount(y, minlength=K).astype(np.float32)
        self._prior_ = (cnt / N).tolist()

        lp  = np.log(cnt / N + 1e-8).astype(np.float32); lp -= lp.mean()
        self.F_init_ = lp.tolist()

        Fsc   = np.tile(lp, (N, 1))
        F_val = np.tile(lp, (X_val.shape[0], 1)) if X_val is not None else None

        oh = np.zeros((N, K), dtype=np.float32)
        oh[np.arange(N), y] = 1.0

        rng = np.random.default_rng(seed)

        best_val_loss = float("inf")
        best_trees:   list = []
        no_improv = 0

        self.trees_: list = []

        ctx = BasicDContext(X, D_num=D_num)
        G_w = np.empty((N, K), dtype=np.float32)
        H_w = np.empty((N, K), dtype=np.float32)
        full_idx = np.arange(N, dtype=np.int32)
        try:
            for m in range(self.n_estimators):
                update_gradients(Fsc, oh, G_w, H_w)

                if self.subsample < 1.0:
                    tree_sub = np.flatnonzero(
                        rng.random(N) < self.subsample
                    ).astype(np.int32)
                    if len(tree_sub) < min(N, 1000):
                        tree_sub = full_idx
                else:
                    tree_sub = full_idx

                t, out_pred = ctx.build(
                    G_w, H_w, tree_sub, self.max_depth, self.reg_lambda
                )

                self.trees_.append(t)
                Fsc += self.learning_rate * out_pred

                val_str = ""
                if X_val is not None:
                    pred_val = t.predict(X_val)
                    F_val    = F_val + self.learning_rate * pred_val
                    Fv_sh    = F_val - F_val.max(axis=1, keepdims=True)
                    P_val    = np.exp(Fv_sh); P_val /= P_val.sum(axis=1, keepdims=True)
                    val_loss = float(
                        -np.log(P_val[np.arange(len(y_val)), y_val].clip(1e-8)).mean()
                    )
                    val_acc  = (P_val.argmax(axis=1) == y_val).mean()
                    val_str  = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improv     = 0
                        best_trees    = list(self.trees_)
                    else:
                        no_improv += 1

                if self.verbose:
                    Fsc_sh = Fsc - Fsc.max(axis=1, keepdims=True)
                    Pm = np.exp(Fsc_sh)
                    Pm /= Pm.sum(axis=1, keepdims=True)
                    ll  = -np.log(Pm[np.arange(N), y].clip(1e-8)).mean()
                    acc = (Pm.argmax(axis=1) == y).mean()
                    print(
                        f"  [BasicDT] Round {m+1:3d} | Loss={ll:.4f} | "
                        f"Acc={acc:.4f}{val_str}"
                    )

                if X_val is not None and self.early_stopping_rounds is not None:
                    if no_improv >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"  [BasicDT] Early stopping at round {m+1}")
                        self.trees_ = best_trees
                        break
        finally:
            ctx.close()
