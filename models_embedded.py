from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

# ---------- pipeline ----------
pipe_embedded = Pipeline([
    ("efs", SelectFromModel(                      # embedded feature selector
        LGBMRegressor(random_state=42,verbosity = -1),
        threshold="median"
    )),
    ("lgb_params_layer1", LGBMRegressor(random_state=42,verbosity = -1))       # final LightGBM regressor
])


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
    
pipe_embedded = SilentPipeline([
    ("efs", SelectFromModel(                      # embedded feature selector
        LGBMRegressor(random_state=42,verbosity = -1),
        threshold="median"
    )),
    ("lgb_params_layer1", LGBMRegressor(random_state=42,verbosity = -1))       # final LightGBM regressor
])

# ---------- hyper-parameter search space ----------
# param_grid_embedded = {
#     # selector-level
#     "efs__threshold":           ["median", "mean", 0.0],
#     "efs__estimator__num_leaves": [31, 63, 127],

#     # final LightGBM
#     "lgb__num_leaves":          [31, 63, 127],
#     "lgb__learning_rate":       [0.01, 0.05, 0.1],
#     "lgb__n_estimators":        [200, 500]
# }

# search = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error",verbose=4)
# search.fit(X_train, y_train)

# y_pred = search.predict(X_test)

# # {'efs__estimator__num_leaves': 31,
# #  'efs__threshold': 'mean',
# #  'lgb__learning_rate': 0.05,
# #  'lgb__n_estimators': 500,
# #  'lgb__num_leaves': 31}

# # X_train_selected = search.best_estimator_.named_steps["efs"].transform(X_train)