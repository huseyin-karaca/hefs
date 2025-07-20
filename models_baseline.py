from typing import Mapping, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import lightgbm as lgb
import numpy as np


class WrappedLGBMRegressor(BaseEstimator, RegressorMixin):
    """
    LightGBM’i doğrudan kullanır ama Grid/RandomSearch’te
    `lgb_params_layer1__learning_rate` benzeri anahtarları kabul eder.
    """

    def __init__(
        self,
        lgb_params_layer1: Mapping[str, Any] | None = {},
        random_state: int | None = None,
    ):
        self.lgb_params_layer1 = lgb_params_layer1        # **dokunma, clone uyumlu**
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, multi_output=False)
        params = self.lgb_params_layer1 or {}
        self._model = lgb.LGBMRegressor(random_state=self.random_state, **params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "_model")
        X = check_array(X, dtype=float)
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # sklearn parametre arayüzü
    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        out = {
            "lgb_params_layer1": self.lgb_params_layer1,
            "random_state": self.random_state,
        }
        if deep and self.lgb_params_layer1:
            out.update({f"lgb_params_layer1__{k}": v for k, v in self.lgb_params_layer1.items()})
        return out

    def set_params(self, **params):
        # genel anahtarlar
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        if "lgb_params_layer1" in params:
            self.lgb_params_layer1 = params.pop("lgb_params_layer1")

        # LightGBM alt parametreleri
        for full_key, val in params.items():
            if full_key.startswith("lgb_params_layer1__"):
                subkey = full_key.split("__", 1)[1]
                if self.lgb_params_layer1 is None:
                    self.lgb_params_layer1 = {}
                self.lgb_params_layer1[subkey] = val
            else:           # beklenmedik anahtar → öznitelik olarak ekle
                setattr(self, full_key, val)
        return self
