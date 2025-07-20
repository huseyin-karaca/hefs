import numpy as np
from sklearn.base   import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing    import MinMaxScaler
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

param_grid_hefs3 = {
    # katman-1  (indeks 0)
    "lgb_params_list__0__learning_rate": [0.01, 0.05, 0.1],
    "lgb_params_list__0__n_estimators":  [200, 400],

    # katman-2  (indeks 1)
    "lgb_params_list__1__num_leaves":    [31, 63, 127],
    "lgb_params_list__1__max_depth":     [3, 5],

    # katman-3  (indeks 2)
    "lgb_params_list__2__learning_rate": [0.01, 0.03],
}

from sklearn.model_selection import GridSearchCV
# gscv = GridSearchCV(
#     estimator= HierarchicalLGBM_K3(
#         splits=[10, 12, 14],
#         lgb_params_list=[
#         {"verbosity": -1},    # layer-1
#         {"verbosity": -1},    # layer-2
#         {"verbosity": -1},    # layer-3
#     ],
#         beta=0.25,              # |α| ≤ 0.25
#         random_state=42),
#     param_grid=param_grid,
#     cv=3,
#     scoring="neg_mean_squared_error",
#     verbose=2,
#     n_jobs=-1
# )
# gscv.fit(X_train, y_train)



class HierarchicalLGBM_K3(BaseEstimator, RegressorMixin):
    """
    K-katmanlı LightGBM hiyerarşisi (yalnız NumPy).

    Parameters
    ----------
    splits : tuple[int]
        Sütun dilimleri. Örn. (p,d,q)  ⇒  ilk p → layer-1, sonraki d → layer-2, son q → layer-3.
    beta : float, default=0.33
        |α| en çok beta olur (0<beta≤1).
    scale_X, scale_y : bool, default=True
        Min-max ölçekleyicileri kullan.
    random_state : int or None

    Üç katmanlı LightGBM hiyerarşisi.
      • Layer-1 → splits[0] kolon
      • Layer-2 → splits[1] kolon (+ ŷ¹)
      • Layer-3 → splits[2] kolon (+ ŷ²)
    """
    # ─────────────────────────────────────────────────────────

    def __init__(
        self,
        splits: tuple[int, int, int],               # ← artık tuple!
        lgb_params_layer1: dict | None = {},
        lgb_params_layer2: dict | None = {},
        lgb_params_layer3: dict | None = {},
        beta: float = 0.33,
        scale_X: bool = True,
        scale_y: bool = True,
        random_state: int | None = None,
    ):
        if not isinstance(splits, tuple) or len(splits) != 3:
            raise ValueError(
                "splits must be a 3-element tuple, e.g. (p, d, q)")
        self.splits = splits
        self.lgb_params_layer1 = lgb_params_layer1
        self.lgb_params_layer2 = lgb_params_layer2 
        self.lgb_params_layer3 = lgb_params_layer3
        self.beta = beta
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.random_state = random_state

    # ------------ yardımcı: tek tek α bul --------------------------------
    @staticmethod
    def _best_alpha_vec(y, yhat, beta):
        pos = np.linspace(0.0,  beta, 30, dtype=np.float32)
        neg = np.linspace(-beta, 0.0, 30, dtype=np.float32)

        y_s    = y[:-1]
        yh_s   = yhat[:-1]
        choose = y_s < yh_s                        # True → negatif grid

        alphas = np.empty_like(y_s)
        for i in range(len(y_s)):
            grid  = neg if choose[i] else pos
            loss  = np.abs(y_s[i] - (1.0 + grid) * yh_s[i])
            alphas[i] = grid[np.argmin(loss)]
        # ilk örnek için alpha=0
        return np.r_[0.0, alphas].astype(np.float32)

    # ------------ iç yardımcı: X'i dilimlere ayır ------------------------
    def _split_cols(self, X):
        idx = np.cumsum(self.splits)          # tuple → ndarray destekli
        return np.split(X, idx[:-1], axis=1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # --- ölçekleme EKLENTİSİ ---------------------------------
        if self.scale_X:
            self.scaler_X_ = MinMaxScaler().fit(X)
            X = self.scaler_X_.transform(X)
        if self.scale_y:
            self.scaler_y_ = MinMaxScaler().fit(y.reshape(-1, 1))
            y = self.scaler_y_.transform(y.reshape(-1, 1)).ravel()
        # ----------------------------------------------------------

        if sum(self.splits) != X.shape[1]:
            raise ValueError("sum(splits) must equal X.shape[1]") 
        if len(self.splits) != 3:
            raise ValueError("splits must have length 3 for K3 model")

        X_parts = self._split_cols(X)
        params_list = [
            self.lgb_params_layer1 or {},
            self.lgb_params_layer2 or {},
            self.lgb_params_layer3 or {},
        ]
        self.models_ = []
        y_prev = None
        for k, (Xk, prm) in enumerate(zip(X_parts, params_list), start=1):
            target = y if k == 1 else self._best_alpha_vec(y, y_prev, self.beta)
            if k > 1:
                Xk = np.column_stack([Xk, y_prev])
            mdl = lgb.LGBMRegressor(random_state=self.random_state, **prm)
            mdl.fit(Xk, target)
            self.models_.append(mdl)
            pred = mdl.predict(Xk)
            y_prev = pred if k == 1 else (1.0 + pred) * y_prev
        self._yhat_train_ = y_prev
        return self
    # ─────────────────────────────────────────────────────────
    def predict(self, X):
        check_is_fitted(self, "models_")
        X = np.asarray(X, dtype=float)
        if self.scale_X:
            X = self.scaler_X_.transform(X)
        X_parts = self._split_cols(X)
        y_prev = None
        for k, (Xk, mdl) in enumerate(zip(X_parts, self.models_), start=1):
            if k > 1:
                Xk = np.column_stack([Xk, y_prev])
            pred = mdl.predict(Xk)
            y_prev = pred if k == 1 else (1.0 + pred) * y_prev
        if self.scale_y:
            y_prev = self.scaler_y_.inverse_transform(y_prev.reshape(-1, 1)).ravel()
        return y_prev
    # ─────────────────────────────────────────────────────────
    # sklearn parametre arayüzü
    # ─────────────────────────────────────────────────────────
    def get_params(self, deep=True):
        out = {
            "splits": self.splits,
            "beta": self.beta,
            "scale_X": self.scale_X,
            "scale_y": self.scale_y,
            "random_state": self.random_state,
            "lgb_params_layer1": self.lgb_params_layer1,
            "lgb_params_layer2": self.lgb_params_layer2,
            "lgb_params_layer3": self.lgb_params_layer3,
        }
        if deep:
            for idx, d in enumerate(
                [self.lgb_params_layer1, self.lgb_params_layer2, self.lgb_params_layer3], 
                start=1
            ):
                if d:
                    out.update({f"lgb_params_layer{idx}__{k}": v for k, v in d.items()})
        return out

    def set_params(self, **params):
        simple = {
            "splits", "beta", "scale_X", "scale_y", "random_state",
            "lgb_params_layer1", "lgb_params_layer2", "lgb_params_layer3"
        }
        for k in list(params.keys()):          # genel öznitelikler
            if k in simple:
                setattr(self, k, params.pop(k))

        # katman içi LightGBM parametreleri
        for full_key, val in params.items():
            if full_key.startswith("lgb_params_layer"):
                prefix, param_name = full_key.split("__", 1)   # lgb_params_layer2, learning_rate
                idx = int(prefix[-1])                          # 1,2,3
                attr_name = f"lgb_params_layer{idx}"
                d = getattr(self, attr_name)
                if d is None:
                    d = {}
                    setattr(self, attr_name, d)
                d[param_name] = val
            else:
                setattr(self, full_key, val)
        return self