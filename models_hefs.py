import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import warnings 
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy")

from scipy.stats import uniform, randint
param_grid_hefs = {
    # ---- layer 1 (top model) ----
    "lgb_params_layer1__learning_rate": uniform(0.005, 0.03),   # 0.005‒0.035
    "lgb_params_layer1__n_estimators":  randint(200, 800),      # 200‒799
    "lgb_params_layer1__num_leaves":    randint(31, 255),       # 31‒254
    
    # ---- layer 2 (bottom model) ----
    "lgb_params_layer2__learning_rate": uniform(0.005, 0.03),
    "lgb_params_layer2__n_estimators":  randint(200, 800),
    "lgb_params_layer2__num_leaves":    randint(15, 127),
    
    # you can add other high-level knobs too, e.g.:
    "lgb_params_layer1__verbosity":   [-1],
    "lgb_params_layer2__verbosity":   [-1]
}

class HierarchicalLightGBMRegressor(BaseEstimator, RegressorMixin):
    """
    Two-layer LightGBM ensemble with per-timestamp alpha scaling.

    Parameters
    ----------
    feat_split : tuple(int, int)
        Number of features for layer-1 and layer-2, e.g. (p, q).
    """
    def __init__(self,
                 feat_split,
                 lgb_params_layer1={},
                 lgb_params_layer2={},
                 random_state=None,
                 scale_X=True,
                 scale_y=True,
                 beta=None):
        self.feat_split = feat_split          # (p, q)
        self.lgb_params_layer1 = lgb_params_layer1 
        self.lgb_params_layer2 = lgb_params_layer2 
        self.random_state = random_state
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.beta = beta

        self._scaler_X = None
        self._scaler_y = None
        self.model_1 = None
        self.model_2 = None

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_alphas(y_true, y_pred, beta=None):
        if beta:
            alpha_pos = np.linspace(0, beta, 300, dtype=np.float32)
            alpha_neg = np.linspace(-beta, 0.0, 300, dtype=np.float32)
        else:
            alpha_pos = np.linspace(0, 0.33, 30, dtype=np.float32)
            alpha_neg = np.linspace(-0.33, 0.0, 30, dtype=np.float32)

        y_shift      = np.roll(y_true, 1)      # t-1 değerleri
        y_hat_shift  = np.roll(y_pred, 1)

        choose_neg   = y_shift < y_hat_shift
        grid         = np.where(choose_neg[:, None], alpha_neg, alpha_pos)
        mse          = np.abs(y_shift[:, None] - (1 + grid) * y_hat_shift[:, None])
        best_idx     = np.argmin(mse, axis=1)
        return grid[np.arange(len(best_idx)), best_idx]

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        p, q = self.feat_split
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)

        # ---------- scaling ----------
        if self.scale_X:
            self._scaler_X = MinMaxScaler().fit(X)
            X_scaled = self._scaler_X.transform(X)
        else:
            X_scaled = X

        if self.scale_y:
            self._scaler_y = MinMaxScaler().fit(y.reshape(-1, 1))
            y_scaled = self._scaler_y.transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y

        X1 = X_scaled[:, :p]
        X2 = X_scaled[:, p:p + q]

        # ---------- layer 1 ----------
        params1 = dict(random_state=self.random_state, **self.lgb_params_layer1)
        self.model_1 = lgb.LGBMRegressor(**params1)
        self.model_1.fit(X1, y_scaled)

        y_hat1 = self.model_1.predict(X1)

        # ---------- alpha (vector) ----------
        alpha_vec = self._compute_alphas(y_scaled, y_hat1, beta=self.beta)

        # ---------- layer 2 ----------
        X2_plus   = np.column_stack([X2, y_hat1])
        params2   = dict(random_state=self.random_state, **self.lgb_params_layer2)
        self.model_2 = lgb.LGBMRegressor(**params2)
        self.model_2.fit(X2_plus, alpha_vec)

        return self

    def predict(self, X):
        check_is_fitted(self, ["model_1", "model_2"])
        p, q = self.feat_split
        X = check_array(X, dtype=np.float32)

        if self.scale_X:
            X_scaled = self._scaler_X.transform(X)
        else:
            X_scaled = X

        X1 = X_scaled[:, :p]
        y_hat1 = self.model_1.predict(X1)

        X2_plus = np.column_stack([X_scaled[:, p:p + q], y_hat1])
        alpha_hat = self.model_2.predict(X2_plus)

        y_pred_scaled = (1 + alpha_hat) * y_hat1

        if self.scale_y:
            y_pred = self._scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled
        return y_pred

    # ---------------------------------------------------------------------
    # >>> THIS is the critical bit <<<
    # ---------------------------------------------------------------------
    def get_params(self, deep=True):
        params = {
            "lgb_params_layer1": self.lgb_params_layer1,
            "lgb_params_layer2": self.lgb_params_layer2,
            "random_state": self.random_state,
            "feat_split": self.feat_split,
        }

        if deep:
            params.update(
                {f"lgb_params_layer1__{k}": v
                 for k, v in self.lgb_params_layer1.items()}
            )
            params.update(
                {f"lgb_params_layer2__{k}": v
                 for k, v in self.lgb_params_layer2.items()}
            )
        return params

    def set_params(self, **params):
        # Pull out any nested LightGBM keys first ---------------
        l1_updates, l2_updates = {}, {}
        keep_for_super = {}
        for k, v in params.items():
            if k.startswith("lgb_params_layer1__"):
                l1_updates[k.split("__", 1)[1]] = v
            elif k.startswith("lgb_params_layer2__"):
                l2_updates[k.split("__", 1)[1]] = v
            else:
                keep_for_super[k] = v

    
        # ---- safely update dicts ---------------------------------
        if l1_updates:
            if self.lgb_params_layer1 is None:          # <-- add this guard
                self.lgb_params_layer1 = {}
            self.lgb_params_layer1.update(l1_updates)

        if l2_updates:
            if self.lgb_params_layer2 is None:          # <-- and this
                self.lgb_params_layer2 = {}
            self.lgb_params_layer2.update(l2_updates)

        for k, v in keep_for_super.items():
            setattr(self, k, v)
        return self
