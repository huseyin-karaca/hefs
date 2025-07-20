from models_embedded import pipe_embedded
from models_ablation import ResidualBoostLGBM
from models_ensemble import ColumnSplitLightGBMRegressor
from models_hefs import HierarchicalLightGBMRegressor
from models_hefs3 import HierarchicalLGBM_K3
from models_wrapper import pipe_wrapper_forward, pipe_wrapper_backward
from models_baseline import WrappedLGBMRegressor
from models_filter import pipe_filter
from lightgbm import LGBMRegressor 

from sklearn.feature_selection import f_regression, mutual_info_regression
from mlforecast import MLForecast 
import numpy as np
import pandas as pd
from hus_helper import load_synt, load_m4, load_etth2, load_weather, report_results, report_results2
from load_weather2 import load_weather2
import json
from sklearn.utils.validation import check_X_y

from utilsforecast.losses import *
from utilsforecast.evaluation import evaluate




# DATA


# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# X, y = make_regression(n_samples=500, n_features=36, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

#X_train, X_test, y_train, y_test = load_synt()
X_train, X_test, y_train, y_test, *_ = load_weather2(300,42)


# X_train, y_train = check_X_y(X_train,y_train)
# X_test, y_test = check_X_y(X_test,y_test)


# MODELS 

model_names_rs_lgb = ["model_hefs","model_hefs3","model_ensemble","model_ablation","pipe_embedded","model_baseline"]
model_names_rs_lgb = ["pipe_wrapper_backward","pipe_wrapper_forward"]
model_names_rs_lgb = ["pipe_filter"]

model_names = [
    "model_hefs",
    "model_hefs3",
    "model_ensemble",
    "model_ablation",
    "pipe_embedded",
    "model_baseline",
    "pipe_filter",
    "pipe_wrapper_forward",
    "pipe_wrapper_backward",
    ]


(p, q) = (10,30)
(pp,qq,zz) = (10,10,20)
num_feat = 40
start_seed = 42
num_seed = 30

hpt = 0
dataset = "weather2"

model_ablation = ResidualBoostLGBM(
    split=p,
    random_state=42,
    scale_X=False,
    scale_y=False)

model_ensemble = ColumnSplitLightGBMRegressor(
    num_feat_second_layer=q,
    alpha_grid=30,
    random_state=42)

model_hefs = HierarchicalLightGBMRegressor(
    feat_split=(p, q),
    random_state=42,
    beta=0.1,
    scale_X=False,
    scale_y=False)

model_hefs3 = HierarchicalLGBM_K3(
    splits=(pp, qq, zz),
    random_state=42,
    scale_X=False,
    scale_y=False)

model_baseline = WrappedLGBMRegressor(
    random_state=42)





if hpt:
### param spaces ######## üç modelimizin search space'i aynı zatennn ##########

    from scipy.stats import uniform, randint, loguniform

    param_grid = {
        # ── Layer-1 (top) ──────────────────────────────────────────────
        "lgb_params_layer1__learning_rate":      loguniform(1e-3, 0.2),
        "lgb_params_layer1__n_estimators":       randint(400, 1500),
        "lgb_params_layer1__num_leaves":         randint(31, 512),
        "lgb_params_layer1__max_depth":          randint(-1, 16),   # -1=no limit
        "lgb_params_layer1__min_child_samples":  randint(5, 100),
        "lgb_params_layer1__min_child_weight":   loguniform(1e-3, 10),
        "lgb_params_layer1__subsample":          uniform(0.5, 0.5), # 0.5–1.0
        "lgb_params_layer1__subsample_freq":     randint(0, 8),
        "lgb_params_layer1__colsample_bytree":   uniform(0.5, 0.5),
        "lgb_params_layer1__lambda_l1":          loguniform(1e-4, 10),
        "lgb_params_layer1__lambda_l2":          loguniform(1e-4, 10),
        "lgb_params_layer1__max_bin":            randint(128, 512),
        "lgb_params_layer1__verbosity":          [-1],
        "lgb_params_layer1__boosting_type":      ["gbdt"],          # or "dart"
        "lgb_params_layer1__objective":          ["mae","mse"],

        # ── Layer-2 (bottom) ───────────────────────────────────────────
        "lgb_params_layer2__learning_rate":      loguniform(1e-3, 0.2),
        "lgb_params_layer2__n_estimators":       randint(400, 1500),
        "lgb_params_layer2__num_leaves":         randint(15, 255),
        "lgb_params_layer2__max_depth":          randint(-1, 16),
        "lgb_params_layer2__min_child_samples":  randint(5, 100),
        "lgb_params_layer2__min_child_weight":   loguniform(1e-3, 10),
        "lgb_params_layer2__subsample":          uniform(0.5, 0.5),
        "lgb_params_layer2__subsample_freq":     randint(0, 8),
        "lgb_params_layer2__colsample_bytree":   uniform(0.5, 0.5),
        "lgb_params_layer2__lambda_l1":          loguniform(1e-4, 10),
        "lgb_params_layer2__lambda_l2":          loguniform(1e-4, 10),
        "lgb_params_layer2__max_bin":            randint(128, 512),
        "lgb_params_layer2__verbosity":          [-1],
        "lgb_params_layer2__boosting_type":      ["gbdt"],
        "lgb_params_layer2__objective":          ["mae","mse"],
        
        # ── Layer-3 (if any) ───────────────────────────────────────────
        "lgb_params_layer3__learning_rate":      loguniform(1e-3, 0.2),
        "lgb_params_layer3__n_estimators":       randint(400, 1500),
        "lgb_params_layer3__num_leaves":         randint(15, 255),
        "lgb_params_layer3__max_depth":          randint(-1, 16),
        "lgb_params_layer3__min_child_samples":  randint(5, 100),
        "lgb_params_layer3__min_child_weight":   loguniform(1e-3, 10),
        "lgb_params_layer3__subsample":          uniform(0.5, 0.5),
        "lgb_params_layer3__subsample_freq":     randint(0, 8),
        "lgb_params_layer3__colsample_bytree":   uniform(0.5, 0.5),
        "lgb_params_layer3__lambda_l1":          loguniform(1e-4, 10),
        "lgb_params_layer3__lambda_l2":          loguniform(1e-4, 10),
        "lgb_params_layer3__max_bin":            randint(128, 512),
        "lgb_params_layer3__verbosity":          [-1],
        "lgb_params_layer3__boosting_type":      ["gbdt"],
        "lgb_params_layer3__objective":          ["mae","mse"],

        # ── Optional high-level knobs (for the wrapper itself) ─────────
        #
        "beta":        uniform(0.05, 1),   # example: search β range
        # "random_state": [0, 42],              # if you want to tune that too

        # ------ Filter (SelectKBest) ----------------------

        'filter__k': randint(5, 25),
        # 'filter__score_func': [
        #     f_regression,           # F-istatistiği
        #     mutual_info_regression  # Karşılıklı bilgi
        # ],

        # ── Embedded Feature Selector (SelectFromModel)-----─────────
        "efs__threshold": uniform(0.5, 1.0),
        "efs__max_features": randint(5, 30),             

        # ───────── SequentialFeatureSelector  (sfs__*) ──────────────
        "sfs__direction": ["forward", "backward"],
        "sfs__n_features_to_select": randint(5, 30),

         # others
        # "feat_split": [(p, num_feat - p) for p in range(5, 20)],
        # "split": range(5,20),
        # "num_feat_second_layer": range(20, num_feat),
        # "splits": [
        #     (pp, qq, num_feat - pp - qq)
        #     for pp in range(5, 15)
        #     for qq in range(5, 15)
        #     if pp + qq < num_feat          # zz ≥ 1 olsun
        # ]

    }



    from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit






    from tqdm import tqdm
    #from tqdm.auto import tqdm
    from joblib import parallel_backend
    import joblib

    # tqdm → joblib entegrasyonu
    from tqdm_joblib import tqdm_joblib




    fit_times = pd.DataFrame(columns = ["mean_fit_time","std_fit_time"],index=model_names_rs_lgb)
    for model_name in model_names_rs_lgb:
        print(f"{model_name} search started...")
        model = eval(model_name)
        

        cv_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            #random_state=0,
            verbose=1
        )

        with tqdm_joblib(tqdm(desc="RandomSearch", total=cv_search.n_iter * cv_search.cv.n_splits)):
            cv_search.fit(X_train, y_train)


        
        # best_model = cv_search.best_estimator_
        import joblib
        

        import json, pandas as pd, joblib
        from pathlib import Path

        RESULTS_DIR = Path("checkpoints")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------
        # after you have done:  search.fit(X_train, y_train)
        # ------------------------------------------
        best_params  = cv_search.best_params_                 # plain dict
        cv_results   = pd.DataFrame(cv_search.cv_results_)                  # dict of arrays


        # 1) save best_params_ as JSON ----------------
        with open(RESULTS_DIR / f"best_params_{dataset}_{model_name}.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # 2) save cv_results_ as CSV -------------------
        cv_results.to_csv(
            RESULTS_DIR / f"cv_results_{dataset}_{model_name}.csv",
            index=False
        )

        # 3) all file
        joblib.dump(cv_search, RESULTS_DIR / f'rs_{dataset}_{model_name}.pkl')


        # 4) fit times

        fit_times.loc[model_name, :] = cv_results[cv_results["rank_test_score"] == 1].filter(
            like="fit_time").values
    fit_times.to_csv(f"fittimes_{dataset}.csv",
        index=True
    )










hpt = 0




if not hpt:
    # model training for results

    
    maes = pd.DataFrame(columns = model_names)
    maes_train = pd.DataFrame(columns = model_names)

    for seed in range(start_seed, start_seed + num_seed):
        print(f"Seed {seed}")
        preds = {}
        preds_train = {}
        X_train, X_test, y_train, y_test, *_ = load_weather2(300,seed)

        for model_name in model_names:
            print(f"{model_name} is fitting...")
            
            model = eval(model_name)

            with open(f"/Users/huseyinkaraca/Desktop/aysin_reproduce/hierarchical-feature-selection/paper/checkpoints/best_params_{dataset}_{model_name}.json") as f:
                best_params = json.load(f)

            model.set_params(**best_params)
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            pred_train = model.predict(X_train)

            preds[model_name] = pred
            preds_train[model_name] = pred_train


        results = pd.DataFrame(preds)
        results["y"] = y_test
        results["unique_id"] = 1
        results["ds"] = results.index + 1

        results_train = pd.DataFrame(preds_train)
        results_train["y"] = y_train
        results_train["unique_id"] = 1
        results_train["ds"] = results_train.index + 1

        from hus_helper import cumsum_plot
        cumsum_plot(results,plot_text= None)

        summary = evaluate(
            results,
            metrics=[mae, mse, rmse, mape, smape]
        ).drop(columns='unique_id').groupby('metric').mean().reset_index()

        summary_train = evaluate(
            results_train,
            metrics=[mae, mse, rmse, mape, smape]
        ).drop(columns='unique_id').groupby('metric').mean().reset_index()


        maes.loc[seed] = summary.iloc[0,1:]
        maes_train.loc[seed] = summary_train.iloc[0,1:]

    scores, tresults = report_results2(maes, maes_train, dataset = dataset, n_train=240, n_test=60)


