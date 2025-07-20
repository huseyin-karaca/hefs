from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (SelectKBest,
                                       f_regression, mutual_info_regression, r_regression)
from lightgbm import LGBMRegressor

class SilentPipeline(Pipeline):
    """Bilinmeyen 'adım__param' anahtarlarını sessizce yok sayar."""
    def set_params(self, **params):
        # hangi step adları mevcut?
        step_names = set(self.named_steps)
        # filtrele
        valid = {k: v for k, v in params.items()
                 if k.split("__", 1)[0] in step_names}
        # standart davranışla kalanları ayarla
        super().set_params(**valid)
        return self


pipe_filter = SilentPipeline([
    ('filter', SelectKBest(score_func = r_regression)),
    ('lgb_params_layer1',    LGBMRegressor(random_state=42, verbosity=-1))
])

