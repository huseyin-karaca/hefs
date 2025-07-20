def load_weather2(N, seed):

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


        # for lag in range(1,25):                         # hourly example
        #     X_others[f'y_lag{lag}'] = y.shift(lag)


        # final learning frame (drop rows with NaNs created by shift)
        dataset = pd.concat([y, X_others], axis=1).dropna()
        dataset.reset_index(inplace = True)
        dataset["unique_id"] = 1
        dataset = dataset.rename(columns={target_id:"y"})
        dataset["ds"] = dataset['ds'].astype('datetime64[ns]')
        X_df["ds"] = X_df['ds'].astype('datetime64[ns]')
        dataset = dataset[["ds","unique_id","y", *dataset.columns.drop(["ds","unique_id","y"])]]
        dataset

        xx = X_df[X_df["unique_id"] == target_id]
        xx.loc[:,"ds"] = xx["ds"].astype("datetime64[ns]")
        xx.loc[:,"unique_id"] = 1
        xx


        dataset = dataset.merge(xx, on = ["unique_id","ds"])
        dataset
        window_sizes = [6, 144, 1008, 4320]  # 1saat, 1gün, 1hafta, 1ay
        shifts = [1, 6, 144, 1008]  # 10dk, 1saat, 1gün, 1hafta
        lags = list(range(1, 13))        # ilk 2 saat (12*10dk)
        lags += [24, 48, 72, 144]        # 4saat, 8saat, 12saat, 24saat
        lags += [288, 720, 1008, 2016, 4320]  # 2gün, 5gün, 1hf, 2hf, 1ay



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
        freq = "10T",
        **my_init
        )
        dataset = mlf.preprocess(dataset,static_features=[])

        dataset = dataset.sample(n = N, random_state = seed)

        X= dataset.drop(columns=["ds","unique_id","y"])
        y = dataset["y"]
        X
        import pandas as pd
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

        # 3) korelasyon‐tabanlı skorlar
        selector = SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train, y_train)

        scores = pd.Series(selector.scores_, index=X_train.columns)
        ranked = scores.sort_values(ascending=False)

        # print(ranked.head(15))      # en yüksekten düşüğe ilk 15 özellik

        X_train = pd.DataFrame(selector.transform(X_train), columns = ranked.index)
        X_test  = pd.DataFrame(selector.transform(X_test), columns = ranked.index)
        X_train

        X_train = X_train.iloc[:,:40]
        X_test = X_test.iloc[:,:40]



        from sklearn.utils.validation import check_X_y
        X_train, y_train = check_X_y(X_train,y_train)
        X_test, y_test = check_X_y(X_test,y_test)

        return X_train, X_test, y_train, y_test
