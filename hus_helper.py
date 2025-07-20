import pandas as pd 
def load_synt():

    X_train = pd.read_csv("/Users/huseyinkaraca/Desktop/aysin_reproduce/X_train_synt.csv",index_col = 0)
    X_test = pd.read_csv("/Users/huseyinkaraca/Desktop/aysin_reproduce/X_test_synt.csv",index_col = 0)
    y_train = pd.read_csv("/Users/huseyinkaraca/Desktop/aysin_reproduce/y_train_synt.csv",index_col = 0).to_numpy().ravel()
    y_test = pd.read_csv("/Users/huseyinkaraca/Desktop/aysin_reproduce/y_test_synt.csv",index_col = 0).to_numpy().ravel()

    return X_train, X_test, y_train, y_test

def load_m4(N = 500, seed = 2, num_feat = 30):
    from datasetsforecast.m4 import M4
    from mlforecast.lag_transforms import RollingStd, RollingMean
    from mlforecast import MLForecast

    group = "Yearly"

    y_df, X_df, S_df = M4.load(".",group)

    y_df["y"] = y_df.groupby("unique_id",observed=False)["y"] \
            .transform(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)

    # window_sizes = [4, 12, 24, 36]
    # shifts = [1, 24, 48, 72, 96]
    # lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,24]

    if group == "Weekly":
        window_sizes = [4, 13, 26, 52]         # rolling mean/std
        shifts       = [1, 13, 26, 52]         # hangi lag üstüne rolling uygulanacak
        lags         = list(range(1, 13)) + [26, 52, 104]
    
    elif group == "Hourly":
        window_sizes = [24, 168, 720]                # 1 d, 1 w, 1 mth
        shifts       = [1, 24]                       # 1 h, 1 d
        lags         = list(range(1, 25)) + [168, 336, 720] # 24 h + 1 w, 2 w, 1 mth

    elif group == "Daily":
        window_sizes = [7, 30, 90, 365]              # 1 w, 1 mth, 1 qtr, 1 yr
        shifts       = [1, 7, 30, 365]               # same granularities for roll
        lags         = list(range(1, 32))            \
                    + [60, 90, 182, 365, 730]       # up to 1 mth + seasonal + 2 yr

    elif group == "Monthly":
        window_sizes = [3, 6, 12, 24]                # 1 qtr, ½ yr, 1 yr, 2 yr
        shifts       = [1, 3, 12]                    # 1 mth, 1 qtr, 1 yr
        lags         = list(range(1, 13))            \
                    + [24, 36, 60]                  # 2 yr, 3 yr, 5 yr

    elif group == "Quarterly":
        window_sizes = [4, 8, 16]                    # 1 yr, 2 yr, 4 yr
        shifts       = [1, 4]                        # 1 qtr, 1 yr
        lags         = list(range(1, 20))            # 3 yr, 4 yr, 5 yr

    elif group in ("Yearly", "Annual"):
        window_sizes = [2, 3, 5, 10]                 # 2-,3-,5-,10-yr trends
        shifts       = [1, 2, 5]                     # 1-,2-,5-yr lags
        lags         = list(range(1, 10)) + [10, 20]  # 5 yr + decade + generation


    y_df['ds'] = y_df['ds'].astype(int)
    #X_df['ds'] = X_df['ds'].astype('datetime64[ns]')


    # Get lagging - rolling features easily by mlf.preprocess
    my_init = {
        "lags": lags,
        "lag_transforms": {
            shift: [f(window_size=w)
                    for w in window_sizes for f in (RollingMean, RollingStd)]
            for shift in shifts
        }
    }
    mlf = MLForecast(
        models =[],
        freq = 1,
        **my_init
    )
    lag_rol_features = mlf.preprocess(y_df)


    lag_rol_features = lag_rol_features.sample(n = N, random_state = seed)
    # sıfır ve iki iyi çalıştı. genel olarak hepsi iyi zaten, belki t-test bakılabilir


    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_regression

    # 1) mlforecast çıktı DataFrame’iniz:
    #    varsayalım lag_rol_features = id, ds, y, …feature_cols… şeklinde
    y = lag_rol_features['y']
    X = lag_rol_features.drop(columns=['y', 'unique_id', 'ds'])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # # 2) ilk pencerelerdeki NaN’leri temizle
    # mask = X.notna().all(axis=1)
    # X, y = X[mask], y[mask]

    # 3) korelasyon‐tabanlı skorlar
    selector = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=X_train.columns)
    ranked = scores.sort_values(ascending=False)

    # print(ranked.head(15))      # en yüksekten düşüğe ilk 15 özellik

    X_train = pd.DataFrame(selector.transform(X_train), columns = ranked.index)
    X_test = pd.DataFrame(selector.transform(X_test), columns = ranked.index)



    from sklearn.utils.validation import check_X_y
    X_train, y_train = check_X_y(X_train,y_train)
    X_test, y_test = check_X_y(X_test,y_test)

    # X_train = X_train[:,:num_feat]
    # X_test = X_test[:,:num_feat]

    # just for trying, be careful here!!!
    X_train = np.concatenate((X_train[:,[0]], X_train[:,5:5+num_feat-1]),axis = 1)
    X_test  = np.concatenate((X_test[:,[0]], X_test[:,5:5+num_feat-1]), axis = 1)

        
    X_train, y_train = check_X_y(X_train,y_train)
    X_test, y_test = check_X_y(X_test,y_test)




    return X_train, X_test, y_train, y_test, ranked








def load_etth2(N, seed):
    from datasetsforecast.long_horizon import LongHorizon
    from mlforecast.lag_transforms import RollingStd, RollingMean
    from mlforecast import MLForecast

    y_df, X_df, S_df = LongHorizon.load(".","ETTh2")

    y_df["y"] = y_df.groupby("unique_id",observed=False)["y"] \
            .transform(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)

    # window_sizes = [4, 12, 24, 36]
    # shifts = [1, 24, 48, 72, 96]
    # lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,24]

    window_sizes = [24, 168, 720]        # 1g, 1hf, 2hf, 1ay

    # rolling uygulanacak kaydırmalar
    shifts       = [1, 24]          # 1saat, 1gün, 1hafta, 1ay

    # düz gecikmeler
    lags         = list(range(1, 25))         # son 24 saat
    lags        += [168, 336, 720 ]      # 1hf, 2hf, 1ay, 1yıl



    #y_df['ds'] = y_df['ds'].astype(int)
    y_df['ds'] = y_df['ds'].astype('datetime64[ns]')
    X_df['ds'] = X_df['ds'].astype('datetime64[ns]')


    # Get lagging - rolling features easily by mlf.preprocess
    my_init = {
        "lags": lags,
        "lag_transforms": {
            shift: [f(window_size=w)
                    for w in window_sizes for f in (RollingMean, RollingStd)]
            for shift in shifts
        }
    }
    mlf = MLForecast(
        models =[],
        freq = "H",
        **my_init
    )
    lag_rol_features = mlf.preprocess(y_df)

    lag_rol_features
    lag_rol_features = lag_rol_features.merge(X_df, on = ["unique_id","ds"])
    lag_rol_features

    lag_rol_features = lag_rol_features.sample(n = N, random_state = seed)
    # sıfır ve iki iyi çalıştı. genel olarak hepsi iyi zaten, belki t-test bakılabilir


    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_regression

    # 1) mlforecast çıktı DataFrame’iniz:
    #    varsayalım lag_rol_features = id, ds, y, …feature_cols… şeklinde
    y = lag_rol_features['y']
    X = lag_rol_features.drop(columns=['y', 'unique_id', 'ds'])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # # 2) ilk pencerelerdeki NaN’leri temizle
    # mask = X.notna().all(axis=1)
    # X, y = X[mask], y[mask]

    # 3) korelasyon‐tabanlı skorlar
    selector = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=X_train.columns)
    ranked = scores.sort_values(ascending=False)

    # print(ranked.head(15))      # en yüksekten düşüğe ilk 15 özellik

    X_train = pd.DataFrame(selector.transform(X_train), columns = ranked.index)
    X_test = pd.DataFrame(selector.transform(X_test), columns = ranked.index)

    X_train = X_train.iloc[:,:40]
    X_test = X_test.iloc[:,:40]



    from sklearn.utils.validation import check_X_y
    X_train, y_train = check_X_y(X_train,y_train)
    X_test, y_test = check_X_y(X_test,y_test)


    return X_train, X_test, y_train, y_test


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid
def cumsum_plot(results_df,plot_text):
    """
    This computes the daily MSE, then does a cumulative sum 
    and plots the average cumsum over time.x
    """

    # ÇOK ÖNEMLİ: convert from ds to horizon
    # 1) Önce mutlaka zaman sırasına göre sıralayın
    results_df = results_df.sort_values(['unique_id', 'ds'])

    # 2) Grup içi sıralamaya karşılık gelen bir sayaç ekleyin
    results_df['horizon'] = results_df.groupby('unique_id').cumcount() + 1

    model_names = [
        "LightGBM",
        "Boosted SDT",
        "Boosted SDT w/Q"
    ]
    model_names = results_df.columns.drop(["y","unique_id","ds","horizon"])

    plt.figure(0)
    # plt.figure(1, figsize=(8, 5))
    # plt.figure(2, figsize=(8, 5))

    for model_name in model_names:
        # print(model_name)
        # Squared errors
        results_df[f'{model_name}_sq_err'] = (
            results_df['y'] - results_df[model_name])**2
        results_df[f'{model_name}_abs_err'] = abs(
            results_df['y'] - results_df[model_name])
        results_df[f'{model_name}_smape'] = abs(
            results_df['y'] - results_df[model_name]) / (abs(results_df['y']) + abs(results_df[model_name]))

        # Group by day, average
        daily_mse = results_df.groupby(
            'horizon')[f'{model_name}_sq_err'].mean().sort_index()
        daily_mae = results_df.groupby(
            'horizon')[f'{model_name}_abs_err'].mean().sort_index()
        daily_smape = results_df.groupby(
            'horizon')[f'{model_name}_smape'].mean().sort_index()

        # Cumulative sum
        mse_cumsum = daily_mse.cumsum()
        smape_cumsum = daily_smape.cumsum()
        mae_cumsum = daily_mae.cumsum()

        # Average cumsum up to day i
        n = np.arange(1, len(mse_cumsum)+1)
        cumavg_mse = mse_cumsum / n
        cumavg_smape = smape_cumsum / n
        cumavg_mae = mae_cumsum / n

        #plt.figure(0)
        #plt.plot(n, cumavg_mse, label=model_name)
        plt.plot(n, cumavg_mae, label=model_name)

        # plt.figure(1,figsize=(8,5))
        # plt.plot(n, daily_mse, label=model_name)

    

    #plt.figure(0, figsize=(8, 5))
    plt.title("$Cumulative$ $MAE_t$\n (All series averaged)")
    plt.xlabel("t")
    plt.ylabel("cMAE")
    # plt.xlim([1, 48])
    # plt.ylim([0,5e6])
    # plt.xticks([12,24,36,48])
    plt.grid(True)
    plt.legend()

    xmin, xmax, ymin, ymax = plt.axis()

    #plt.text(1.02* xmax,ymin,plot_text,fontsize = 7)

    plt.savefig("cumsum_plots/" + uuid.uuid4().hex + ".pdf", bbox_inches="tight")
    plt.savefig("cumsum_plots/last.pdf", bbox_inches="tight")
    plt.close()



def load_weather(N, seed):
    from datasetsforecast.long_horizon import LongHorizon
    from mlforecast.lag_transforms import RollingStd, RollingMean
    from mlforecast import MLForecast
    import pandas as pd

    y_df, X_df, S_df = LongHorizon.load(".","Weather")

    y_df["y"] = y_df.groupby("unique_id",observed=False)["y"] \
            .transform(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)

    wide = y_df.pivot(index='ds', columns='unique_id', values='y')

    target_id = "T (degC)"
    y        = wide[[target_id]]                       # label
    X_others = wide.drop(columns=target_id)          # each remaining id is a feature

    # (optional) give columns a prefix so you don’t confuse them with lags later
    X_others = X_others.add_prefix('peer_')


    for lag in range(1,25):                         # hourly example
        X_others[f'y_lag{lag}'] = y.shift(lag)

    # final learning frame (drop rows with NaNs created by shift)
    dataset = pd.concat([y, X_others], axis=1).dropna()
    dataset

    # window_sizes = [24, 168, 720]        # 1g, 1hf, 2hf, 1ay

    # # rolling uygulanacak kaydırmalar
    # shifts       = [1, 24]          # 1saat, 1gün, 1hafta, 1ay

    # # düz gecikmeler
    # lags         = list(range(1, 25))         # son 24 saat
    # lags        += [168, 336, 720 ]      # 1hf, 2hf, 1ay, 1yıl



    #y_df['ds'] = y_df['ds'].astype(int)
    # y_df['ds'] = y_df['ds'].astype('datetime64[ns]')
    # X_df['ds'] = X_df['ds'].astype('datetime64[ns]')


    # # Get lagging - rolling features easily by mlf.preprocess
    # my_init = {
    #     "lags": lags,
    #     "lag_transforms": {
    #         shift: [f(window_size=w)
    #                 for w in window_sizes for f in (RollingMean, RollingStd)]
    #         for shift in shifts
    #     }
    # }
    # mlf = MLForecast(
    #     models =[],
    #     freq = "H",
    #     **my_init
    # )
    # lag_rol_features = mlf.preprocess(y_df)

    # lag_rol_features
    # lag_rol_features = lag_rol_features.merge(X_df, on = ["unique_id","ds"])
    # lag_rol_features

    # lag_rol_features = lag_rol_features.sample(n = 1000, random_state = 2)
    # sıfır ve iki iyi çalıştı. genel olarak hepsi iyi zaten, belki t-test bakılabilir


    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_regression

    # # 1) mlforecast çıktı DataFrame’iniz:
    # #    varsayalım lag_rol_features = id, ds, y, …feature_cols… şeklinde
    # y = lag_rol_features['y']
    # X = lag_rol_features.drop(columns=['y', 'unique_id', 'ds'])
    
    dataset = dataset.sample(n= N, random_state=seed)
    X = dataset[X_others.columns]
    y = dataset[target_id]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # # 2) ilk pencerelerdeki NaN’leri temizle
    # mask = X.notna().all(axis=1)
    # X, y = X[mask], y[mask]

    # 3) korelasyon‐tabanlı skorlar
    selector = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=X_train.columns)
    ranked = scores.sort_values(ascending=False)

    # print(ranked.head(15))      # en yüksekten düşüğe ilk 15 özellik

    X_train = pd.DataFrame(selector.transform(X_train), columns = ranked.index)
    X_test = pd.DataFrame(selector.transform(X_test), columns = ranked.index)

    # X_train = X_train.iloc[:,10:]
    # X_test = X_test.iloc[:,10:]



    from sklearn.utils.validation import check_X_y
    X_train, y_train = check_X_y(X_train,y_train)
    X_test, y_test = check_X_y(X_test,y_test)


    return X_train, X_test, y_train, y_test, ranked







def report_results(maes, maes_train, n_train, n_test, dataset = "weather", model_name_our = "model_hefs"):
    import numpy as np
    from scipy.stats import t
    import pandas as pd
    from scipy.stats import shapiro



    def corrected_std(differences, n_train, n_test):
        """Corrects standard deviation using Nadeau and Bengio's approach.

        Parameters
        ----------
        differences : ndarray of shape (n_samples,)
            Vector containing the differences in the score metrics of two models.
        n_train : int
            Number of samples in the training set.
        n_test : int
            Number of samples in the testing set.

        Returns
        -------
        corrected_std : float
            Variance-corrected standard deviation of the set of differences.
        """
        # kr = k times r, r times repeated k-fold crossvalidation,
        # kr equals the number of times the model was evaluated
        kr = len(differences)
        corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
        corrected_std = np.sqrt(corrected_var)
        return corrected_std


    def compute_corrected_ttest(differences, df, n_train, n_test):
        """Computes right-tailed paired t-test with corrected variance.

        Parameters
        ----------
        differences : array-like of shape (n_samples,)
            Vector containing the differences in the score metrics of two models.
        df : int
            Degrees of freedom.
        n_train : int
            Number of samples in the training set.
        n_test : int
            Number of samples in the testing set.

        Returns
        -------
        t_stat : float
            Variance-corrected t-statistic.
        p_val : float
            Variance-corrected p-value.
        """
        mean = np.mean(differences)
        std = corrected_std(differences, n_train, n_test)
        t_stat = mean / std
        p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
        return t_stat, p_val
    
    
    model_names_rivals = maes.columns.drop(model_name_our)

    model_1_scores = maes["model_hefs"].values  # scores of the best model

    scores = pd.DataFrame(columns = ["Train","Test"], index=maes.columns)
    scores["Train"] = maes_train.mean()
    scores["Test"] = maes.mean()
    
    tresults  = pd.DataFrame(columns = ["t-statistics","p-value","shapiro"])

    

    for model_name_rival in model_names_rivals:
        model_2_scores = maes[model_name_rival].values  # scores of the second-best model


        differences = model_1_scores - model_2_scores

        n = differences.shape[0]  # number of test sets
        df = n - 1

        t_stat, p_val = compute_corrected_ttest(-differences, df, n_train, n_test) #multiply with - to make it left-tailed ("less hypothesis")
        
        #apply shapiro-wilk test to check normality. p > 0.05 means it can be regarded as normal
        stat_shapiro, p_shapiro = shapiro(differences)
        tresults.loc[model_name_rival] = (t_stat, p_val, p_shapiro)
        

        # print(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")

    scores.to_csv(f"scores_{dataset}.csv")
    tresults.to_csv(f"tresults_{dataset}.csv")
    
    return scores, tresults

def report_results2(
        maes, maes_train, n_train, n_test,
        dataset="weather", model_name_our="model_hefs",
        alpha=0.05
    ):
    """
    Returns two DataFrames

    scores   : mean train/test MAE for every model
    tresults : per–rival model
               ├─ classical paired-t      (t_raw / p_raw)
               ├─ Nadeau-Bengio corrected (t_nb  / p_nb)
               ├─ multiplicity-adjusted   (p_bonf, p_holm, p_fdr)
               └─ Shapiro-Wilk p-value    (p_shapiro)

    → Compare the ‘sig?’ boolean columns to see which correction is toughest.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import t, shapiro
    from statsmodels.stats.multitest import multipletests

    # ---------- helpers --------------------------------------------------
    def nb_std(diffs, n_tr, n_te):
        kr = len(diffs)                       # k×r evaluations
        var_nb = np.var(diffs, ddof=1) * (1/kr + n_te / n_tr)
        return np.sqrt(var_nb)

    def nb_t_p(diffs, df, n_tr, n_te):
        m  = np.mean(diffs)
        se = nb_std(diffs, n_tr, n_te)
        t_ = m / se
        p_ = t.sf(np.abs(t_), df)             # right-tailed
        return t_, p_

    # ---------- score summary --------------------------------------------
    scores = pd.DataFrame(index=maes.columns, columns=["Train", "Test"])
    scores["Train"] = maes_train.mean()
    scores["Test"]  = maes.mean()

    # ---------- pairwise tests -------------------------------------------
    best_scores    = maes[model_name_our].values
    rivals         = maes.columns.drop(model_name_our)

    cols   = ["t_raw", "p_raw",
              "t_nb",  "p_nb",
              "p_bonf", "p_holm", "p_fdr",
              "p_shapiro"]
    tresults = pd.DataFrame(index=rivals, columns=cols, dtype=float)

    for m in rivals:
        diffs = best_scores - maes[m].values          # + ⇒ best is better
        n     = diffs.size
        df    = n - 1

        # 1) classical paired t-test (left-tailed: best < rival ⇒ negative diff)
        mean  = np.mean(-diffs)
        se    = np.std(diffs, ddof=1) / np.sqrt(n)
        t_raw = mean / se
        p_raw = t.sf(np.abs(t_raw), df)               # right-tail on −diffs

        # 2) Nadeau-Bengio corrected
        t_nb, p_nb = nb_t_p(-diffs, df, n_train, n_test)

        # 3) normality check
        _, p_shap = shapiro(diffs)

        # store temporary (multiplicity adj. later)
        tresults.loc[m, ["t_raw", "p_raw",
                         "t_nb",  "p_nb",
                         "p_shapiro"]] = t_raw, p_raw, t_nb, p_nb, p_shap

    # ---------- multiple-testing adjustments -----------------------------
    pvec = tresults["p_nb"].astype(float).values
    # Bonferroni
    p_bonf = np.minimum(pvec * len(pvec), 1.0)
    # Holm (step-down Bonferroni)
    p_holm = multipletests(pvec, alpha=alpha, method='holm')[1]
    # Benjamini–Hochberg FDR
    p_fdr  = multipletests(pvec, alpha=alpha, method='fdr_bh')[1]

    tresults["p_bonf"] = p_bonf
    tresults["p_holm"] = p_holm
    tresults["p_fdr"]  = p_fdr

    # Optional: flag significance at the chosen alpha
    for col in ["p_raw", "p_nb", "p_bonf", "p_holm", "p_fdr"]:
        tresults[f"sig@{col}"] = tresults[col] < alpha

    # ---------- I/O ------------------------------------------------------
    scores.to_csv(f"scores_{dataset}.csv")
    tresults.to_csv(f"tresults_{dataset}.csv")

    return scores, tresults

# utils_m4.py  ── paste this in a file and import where you need it
import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import math
import pandas as pd

def load_m4_hourly_aysin(
    path_train = "/Users/huseyinkaraca/Desktop/aysin_reproduce/hierarchical-feature-selection/paper/m4/datasets/Hourly-train.csv",
    path_test = "/Users/huseyinkaraca/Desktop/aysin_reproduce/hierarchical-feature-selection/paper/m4/datasets/Hourly-test.csv",
    lags      =(1,2,3,4,5,6,7,8,9,10),
    seed      =None,
    N = 300
):

    def add_date_features(y):
        date = pd.date_range(end="2023-01-01", periods=len(y), freq="H")
        df   = pd.DataFrame(y)
        df["day_of_week"]  = date.dayofweek
        df["day_of_month"] = date.day
        df["week_of_year"] = date.isocalendar().week.astype(int).values
        df["quarter_of_year"] = date.quarter

        def sc(c):
            m = c.max()
            return [math.sin(2*math.pi*x/m) for x in c], \
                   [math.cos(2*math.pi*x/m) for x in c]

        df["month"] = date.month - 1
        df["month_sin"],            df["month_cos"]            = sc(date.month-1)
        df["day_of_month_sin"],     df["day_of_month_cos"]     = sc(date.day)
        df["weekday_sin"],          df["weekday_cos"]          = sc(date.weekday)
        df["week_of_year_sin"],     df["week_of_year_cos"]     = sc(date.isocalendar().week.astype(int)-1)
        df["season_sin"],           df["season_cos"]           = sc((date.month % 12)//3)
        df.index = date
        return df.drop(columns=["y"])

    # ---------- load & align ----------
    tr  = pd.read_csv(path_train, index_col=0).T
    te  = pd.read_csv(path_test, index_col=0).T
    tr  = tr.sample(axis="columns", random_state=seed).dropna()
    #tr  = tr.sample(random_state = seed, n=N).dropna()
    te  = te[tr.columns].dropna()

    data = pd.concat([tr, te]).reset_index(drop=True)
    data.columns = ["y"]
    data = data.diff().dropna()

    # ---------- lag/rolling features ----------
    for k in lags:
        shifted = data["y"].shift(k)
        data[f"y_lag{k}_mean"] = shifted.rolling(k+1).mean()
        data[f"y_lag{k}_std"]  = shifted.rolling(k+1).std()
    data = data.dropna()

    dt_feats   = add_date_features(data["y"])
    data.index = dt_feats.index           # sync indices

    X1 = data.drop(columns=["y"])
    X2 = dt_feats
    y  = data["y"]

    n_test   = len(te)
    X_train, X_test = X1.iloc[:-n_test], X1.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test],  y.iloc[-n_test:]
    X_train2, X_test2 = X2.iloc[:-n_test], X2.iloc[-n_test:]

    # return (pd.concat([y, X1, X2], axis=1),   # data_all
    #         X_train, X_test, y_train.to_frame("y"), y_test.to_frame("y"),
    #         X_train2, X_test2)

    sampled = X_train.sample(n = N, random_state = seed).dropna().index

    return X_train.join(X_train2).loc[sampled,:], X_test.join(X_test2), y_train[sampled], y_test



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pandas as pd

def load_synt_make_classification(
        seed = 42,
        phi = np.array([-2.17426479, -1.89012341, -0.57825805]), 
        theta = np.array([ 1.38494367,  0.60792112, -0.12253469]), 
        d = 0, 
        t = 0,
        mu = 0,
        sigma = 1, 
        N = 500,
        start = "09/01/2022 00:00:00", 
        end = "09/21/2022 20:00:00",
        lags = [1,2,3,4,7], 
        n_f = 26,  
        val_ratio=0.2, 
        MA = False,
        class_imbalance = 0.75,
        k = 0.4):
    
    n = 500
    def ARIMA(phi, theta, d, t, mu, sigma, n, burn=5, init=None, MA=MA):
        """ Simulate data from ARMA model (eq. 1.2.4):
        y_t = phi_1*y_{t-1} + ... + phi_p*y_{t-p} + theta_0*epsilon_t + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q}
        with d unit roots for ARIMA model.
        Arguments:
        phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
        theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
        d -- number of unit roots for non-stationary time series
        t -- value deterministic linear trend
        mu -- mean value for normal distribution error term
        sigma -- standard deviation for normal distribution error term
        n -- length time series
        burn -- number of discarded values because series beginns without lagged terms
        Return:
        x -- simulated ARMA process of shape (n, 1)
        """
        # add theta_0 = 1 to theta
        theta = np.append(1, theta)

        # phi -- array of shape (p,)
        p = phi.shape[0]

        # theta -- array of shape (q,1)
        q = theta.shape[0]

        # add error terms (n+q)
        epsilon = np.random.normal(mu, sigma, (n + max(n, q) + burn, 1))

        # create array for returned values
        x = np.zeros((n + max(n, q) + burn, 1))

        # initialize first time series value y_{t-p}=theta_0*epsilon_{t-q}
        x[0] = epsilon[0]

        for i in range(1, x.shape[0]):
            AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
            MA = np.dot(theta[0: min(i + 1, q)], np.flip(epsilon[i - min(i, q - 1): i + 1], 0))
            if MA:
                x[i] = AR + MA + t
            else:
                x[i] = AR + t


        # add unit roots
        if d != 0:
            ARMA = x[-n:]
            m = ARMA.shape[0]
            y = np.zeros((m + 1, 1))  # create temp array

            for i in range(d):
                for j in range(m):
                    y[j + 1] = ARMA[j] + y[j]
                ARMA = y[1:]
            x[-n:] = y[1:]


        return  x[-n:]

    np.random.seed(seed)
    #k = np.random.rand()

    data = pd.DataFrame(ARIMA(phi, theta, d, t,mu,sigma, n,MA))
    # data = pd.read_csv("/Users/huseyinkaraca/Desktop/aysin_reproduce/hierarchical-feature-selection/synthetic_data.csv")

    data.columns = ["y"]
    data.index = pd.date_range(start=start, end=end,tz = None, freq="h")[:-1]

    dataset = make_classification(data.shape[0], n_features = n_f, n_informative = n_f, n_redundant=0, weights = [class_imbalance], random_state = seed)
    col_names = []
    for i in range(n_f):
        col_names.append("x"+str(i))
    dataset_x = pd.DataFrame(dataset[0]).set_index(data.index)
    dataset_y = pd.DataFrame(dataset[1]).set_index(data.index)
    dataset_x.columns = col_names
    dataset_y.columns = ["y"]

    X_train, X_test, _, _ = train_test_split(dataset_x, dataset_y, test_size=val_ratio, random_state=42, shuffle=False)

    data[dataset_y == 1] = data[dataset_y == 1] * (1+k)
    data[dataset_y == 0] = data[dataset_y == 0] *(1-k)
    noise = np.random.normal(0, 0.5, data.shape[0])
    data = data.add(noise, axis=0)
    data = data.dropna()
    for lag in lags:
        # data["y" + '_lag_' + str(lag)] = data["y"].transform(lambda x: x.shift(lag, fill_value=0))
        # data["y" + '_lag_' + str(lag)+"_std"]= data["y" + '_lag_' + str(lag)].std()
        # data["y" + '_lag_' + str(lag) + "_mean"] = data["y" + '_lag_' + str(lag)].mean()
        data["y" + '_lag_' + str(lag) + "_rolling_mean"] = (
            data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).mean()
        data["y" + '_lag_' + str(lag) + "_rolling_std"] = (
            data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).std()
    data = data.dropna()
    data_all = dataset_x.merge(data, left_index =True, right_index = True)
    # plt.figure(figsize=(15, 15))
    # corrmat = data_all.corr()
    # hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 7}, cmap="Spectral_r")
    # plt.title("Correlation Matrix")
    # plt.savefig("synthetic_corr_matrix.png")
    # plt.show()
    second_set = data_all.iloc[:,:n_f]
    first_set = data_all.iloc[:,n_f:]
    X_train, X_test, y_train, y_test = train_test_split(first_set.iloc[:,1:], first_set.iloc[:, 0], test_size=val_ratio,
                                                        random_state=42, shuffle=False)
    X_train2, X_test2, = train_test_split(second_set, test_size=val_ratio,
                                                        random_state=42, shuffle=False)
    y_train = pd.DataFrame(y_train, index=X_train.index, columns=["y"])
    y_test = pd.DataFrame(y_test, index=X_test.index, columns=["y"])

    
    X_train = X_train.join(X_train2)
    X_test = X_test.join(X_test2)


    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)

    X_train = X_train[:N,:]
    y_train = y_train[:N]

    X_test  = X_test[:N,:]
    y_test  = y_test[:N]
    

    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)

    
    # selector = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
    # X_train = selector.transform(X_train) 
    # X_test = selector.transform(X_test)

    

    return X_train, X_test, y_train, y_test



import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y
from sklearn.feature_selection import f_regression, SelectKBest

def load_synt_make_regression(N, seed):

    # ARMA(2,1) with φ = [0.75, -0.25] and θ = [0.65]
    ar = np.r_[1, -0.75, -0.25, 0.1, -0.1]   # note the sign flip: statsmodels expects +1, −φ₁, …, −φ_p
    ma = np.r_[1,  0.35, -0.15, -0.1]
    np.random.seed(seed)
    y1 = arma_generate_sample(ar=ar, ma=ma, nsample=N, scale=1.0, distrvs=np.random.normal)

    # plt.plot(y)


    y1 = MinMaxScaler().fit_transform(y1.reshape(-1,1))
    #plt.plot(y1)

    X2, y2 = make_regression(
        n_samples=N,
        n_features=26,
        n_informative=26,
        random_state=seed,
        shuffle = True,
        )

    y2 = MinMaxScaler().fit_transform(y2.reshape(-1,1))
    #plt.plot(y2)


    def make_lag_rolling_feats(
            y : pd.Series,
            lags        = None,
            shifts      = None,
            window_sizes=None
        ) -> pd.DataFrame:
        """
        Build a feature DataFrame with:
        • plain lags                ->  y[t-k]
        • rolling mean/std features ->  mean/std( y[t-shift-w+1 : t-shift] )
        
        Parameters
        ----------
        y : pd.Series
            Target time series, indexed by timestamp (or just integer index).
        lags : list[int]
            Simple lag offsets (default: 1‥12 plus 26, 52, 104).
        shifts : list[int]
            Base shifts for rolling stats (default: 1, 13, 26, 52).
        window_sizes : list[int]
            Rolling window lengths (default: 4, 13, 26, 52).

        Returns
        -------
        df : pd.DataFrame
            All feature columns, aligned with y (no rows dropped).
        """

        # defaults from your snippet
        # if lags         is None: lags         = list(range(1, 13)) + [26, 52, 104]
        # if shifts       is None: shifts       = [1, 13, 26, 52]
        # if window_sizes is None: window_sizes = [4, 13, 26, 52]

        df = pd.DataFrame(index=y.index)

        # 1) plain lags
        for k in lags:
            df[f"y_lag_{k}"] = y.shift(k)

        # 2) rolling mean & std for every (shift, window) combo
        for s in shifts:
            ys = y.shift(s)
            for w in window_sizes:
                roll = ys.rolling(window=w, min_periods=w)
                df[f"y_shift{s}_roll{w}_mean"] = roll.mean()
                df[f"y_shift{s}_roll{w}_std"]  = roll.std(ddof=0)

        return df.dropna().values


    X1 = make_lag_rolling_feats(pd.Series(y1.squeeze()), lags= range(1,10), shifts=[1], window_sizes=[4,5])

    y = y1 + y2


    X = np.concatenate([X1, X2[-X1.shape[0]:,:]], axis=1)
    y = y[-X1.shape[0]:,:]


    
    X,y = check_X_y(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.8, test_size=0.2, shuffle = False)

    
    # rank features (if we are going to apply split search)

    selector = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)

    # scores = pd.Series(selector.scores_, index=X_train.columns)
    # ranked = scores.sort_values(ascending=False)

    # print(ranked.head(15))      # en yüksekten düşüğe ilk 15 özellik

    # X_train = pd.DataFrame(selector.transform(X_train), columns = ranked.index)
    # X_test = pd.DataFrame(selector.transform(X_test), columns = ranked.index)

    X_train = selector.transform(X_train) 
    X_test = selector.transform(X_test)


    return X_train, X_test, y_train, y_test 
