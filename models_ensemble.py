from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import lightgbm as lgb


class ColumnSplitLightGBMRegressor(BaseEstimator, RegressorMixin):
    """
    Fixed-LightGBM variant of ColumnSplitEnsembleRegressor.

      • Model-1 uses the leading p features.
      • Model-2 uses the trailing q features.
      • Final ŷ = α·ŷ₁ + (1–α)·ŷ₂ with α tuned on a hold-out split.
    """

    def __init__(
        self,
        num_feat_second_layer: int,
        lgb_params_layer1: Mapping[str, Any] | None = {},
        lgb_params_layer2: Mapping[str, Any] | None = {},
        alpha_grid: int | Sequence[float] = 30,
        holdout_size: float = 0.2,
        metric=mean_absolute_error,
        random_state: int | None = 42,
    ):
        self.num_feat_second_layer = num_feat_second_layer
        self.lgb_params_layer1 = lgb_params_layer1
        self.lgb_params_layer2 = lgb_params_layer2
        self.alpha_grid = alpha_grid
        self.holdout_size = holdout_size
        self.metric = metric
        self.random_state = random_state

    # ──────────────────────────────────────────────────────────
    def _split_cols(self, X: np.ndarray):
        q = self.num_feat_second_layer
        if not 0 < q < X.shape[1]:
            raise ValueError(f"num_feat_second_layer must be in [1,{X.shape[1]-1}]")
        return X[:, :-q], X[:, -q:]

    # ──────────────────────────────────────────────────────────
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, multi_output=False)

        X_lead, X_trail = self._split_cols(X)

        Xl_tr, Xl_val, Xt_tr, Xt_val, y_tr, y_val = train_test_split(
            X_lead, X_trail, y,
            test_size=self.holdout_size,
            shuffle=False,
        )

        self.est1_ = lgb.LGBMRegressor(random_state=self.random_state,
                                       **self.lgb_params_layer1).fit(Xl_tr, y_tr)
        self.est2_ = lgb.LGBMRegressor(random_state=self.random_state,
                                       **self.lgb_params_layer2).fit(Xt_tr, y_tr)

        y1_val, y2_val = self.est1_.predict(Xl_val), self.est2_.predict(Xt_val)

        alphas = (np.linspace(0, 1, int(self.alpha_grid))
                  if np.isscalar(self.alpha_grid) else np.asarray(self.alpha_grid))
        losses = [self.metric(y_val, a*y1_val + (1-a)*y2_val) for a in alphas]
        best_idx = int(np.argmin(losses))
        self.alpha_, self.val_loss_ = float(alphas[best_idx]), float(losses[best_idx])

        # refit on full data
        self.est1_.fit(X_lead, y)
        self.est2_.fit(X_trail, y)
        return self

    # ──────────────────────────────────────────────────────────
    def predict(self, X):
        check_is_fitted(self, ["est1_", "est2_", "alpha_"])
        X = check_array(X, dtype=float)
        X_lead, X_trail = self._split_cols(X)
        return (self.alpha_ * self.est1_.predict(X_lead) +
                (1 - self.alpha_) * self.est2_.predict(X_trail))

    # ──────────────────────────────────────────────────────────
    # >>> NESTED PARAMETER HANDLING <<<
    # Makes RandomizedSearchCV see keys like "lgb1__num_leaves"
    def get_params(self, deep=True):
        base = {
            "num_feat_second_layer": self.num_feat_second_layer,
            "alpha_grid": self.alpha_grid,
            "holdout_size": self.holdout_size,
            "metric": self.metric,
            "random_state": self.random_state,
            "lgb_params_layer1": self.lgb_params_layer1,
            "lgb_params_layer2": self.lgb_params_layer2,
        }
        if deep:
            if self.lgb_params_layer1:                  # guard for None
                base.update({f"lgb_params_layer1__{k}": v
                            for k, v in self.lgb_params_layer1.items()})
            if self.lgb_params_layer2:
                base.update({f"lgb_params_layer2__{k}": v
                            for k, v in self.lgb_params_layer2.items()})
        return base
    

    def set_params(self, **params):
        l1_upd, l2_upd, base = {}, {}, {}
        for k, v in params.items():
            if k.startswith("lgb_params_layer1__"):
                l1_upd[k.split("__", 1)[1]] = v
            elif k.startswith("lgb_params_layer2__"):
                l2_upd[k.split("__", 1)[1]] = v
            else:
                base[k] = v

        # ---- safely update dicts ---------------------------------
        if l1_upd:
            if self.lgb_params_layer1 is None:          # <-- add this guard
                self.lgb_params_layer1 = {}
            self.lgb_params_layer1.update(l1_upd)

        if l2_upd:
            if self.lgb_params_layer2 is None:          # <-- and this
                self.lgb_params_layer2 = {}
            self.lgb_params_layer2.update(l2_upd)

        for k, v in base.items():
            setattr(self, k, v)
        return self
