import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

# model = ResidualBoostLGBM(split=10,
#                           random_state=42,
#                           lgb_params_layer1={"verbosity": -1},
#                           lgb_params_layer2={"verbosity": -1})
param_grid_ablation = {
    "lgb_params_layer1__learning_rate": [0.03, 0.05],
    "lgb_params_layer1__n_estimators":  [200, 400],
    "lgb_params_layer2__num_leaves":    [31, 63],
}
# gcv = GridSearchCV(model, grid, cv=3, n_jobs=-1, verbose=4)
# gcv.fit(X_train, y_train)

# y_pred = gcv.predict(X_test)
# print("Best params:", gcv.best_params_)


class ResidualBoostLGBM(BaseEstimator, RegressorMixin):
    """
    K=2 LightGBM residual-boost ensemble (no alpha block).

        X[:, :split]   -> model_1  predicts ŷ¹
        residual r = (y - ŷ¹)
        X[:, split:]   -> model_2  predicts r̂
        final ŷ  = ŷ¹ + r̂
    """

    def __init__(
        self,
        split,                       # int, p özellik 1. katmana
        lgb_params_layer1={},
        lgb_params_layer2={},
        cat_feats = [],
        scale_X=True,
        scale_y=True,
        random_state=None,
    ):
        self.split = split
        self.lgb_params_layer1 = lgb_params_layer1
        self.lgb_params_layer2 = lgb_params_layer2
        self.cat_feats = cat_feats
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.random_state = random_state

    # ----------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, multi_output=False)
        p = self.split
        if p <= 0 or p >= X.shape[1]:
            raise ValueError("split must be in [1, n_features-1]")

        # ölçekleme
        if self.scale_X:
            self.scaler_X_ = MinMaxScaler().fit(X)
            X = self.scaler_X_.transform(X)
        if self.scale_y:
            self.scaler_y_ = MinMaxScaler().fit(y.reshape(-1, 1))
            y = self.scaler_y_.transform(y.reshape(-1, 1)).ravel()

        X1, X2 = X[:, :p], X[:, p:]

        # katman-1
        self.model_1_ = lgb.LGBMRegressor(random_state=self.random_state,
                                          **self.lgb_params_layer1)
        self.model_1_.fit(X1, y)
        y_hat1 = self.model_1_.predict(X1)

        # residual
        residual = y - y_hat1

        # katman-2
        self.model_2_ = lgb.LGBMRegressor(random_state=self.random_state,
                                          **self.lgb_params_layer2)
        self.model_2_.fit(X2, residual)
        return self

    # ----------------------------------------------------------
    def predict(self, X):
        check_is_fitted(self, ["model_1_", "model_2_"])
        X = check_array(X, dtype=float)
        if self.scale_X:
            X = self.scaler_X_.transform(X)

        p = self.split
        X1, X2 = X[:, :p], X[:, p:]
        yhat1 = self.model_1_.predict(X1)
        res_hat = self.model_2_.predict(X2)
        y_pred_scaled = yhat1 + res_hat

        if self.scale_y:
            y_pred = self.scaler_y_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
        else:
            y_pred = y_pred_scaled
        return y_pred

    # ----------------------------------------------------------
    # GridSearch uyumlu ----------------
    def get_params(self, deep=True):
        params = {
            "split": self.split,
            "scale_X": self.scale_X,
            "scale_y": self.scale_y,
            "random_state": self.random_state,
            "lgb_params_layer1": self.lgb_params_layer1,
            "lgb_params_layer2": self.lgb_params_layer2,
        }
        if deep:
            params.update({f"lgb_params_layer1__{k}": v
                           for k, v in self.lgb_params_layer1.items()})
            params.update({f"lgb_params_layer2__{k}": v
                           for k, v in self.lgb_params_layer2.items()})
        return params
    # ----------------------------------------------------------
    def set_params(self, **params):
        l1_upd, l2_upd, base = {}, {}, {}
        for k, v in params.items():
            if k.startswith("lgb_params_layer1__"):
                l1_upd[k.split("__", 1)[1]] = v
            elif k.startswith("lgb_params_layer2__"):
                l2_upd[k.split("__", 1)[1]] = v
            else:
                base[k] = v

        # ---- güvenli güncelleme ---------------------------------
        if l1_upd:
            # henüz bir dict yoksa oluştur
            if self.lgb_params_layer1 is None:
                self.lgb_params_layer1 = {}
            self.lgb_params_layer1.update(l1_upd)

        if l2_upd:
            if self.lgb_params_layer2 is None:
                self.lgb_params_layer2 = {}
            self.lgb_params_layer2.update(l2_upd)

        for k, v in base.items():
            setattr(self, k, v)
        return self
