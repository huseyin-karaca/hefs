# Run HEFS

<p class="lede">Try the two-layer estimator, explore the experiment setup, and find the datasets used in the paper.</p>

## Quickstart

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/quickstart.py
```

On Windows, activate with `.venv\Scripts\activate`. LightGBM may require an OpenMP runtime; follow its [installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) if the native library cannot load.

The example generates a time-ordered regression problem with two predefined feature groups, fits HEFS, and prints held-out MAE alongside a single LightGBM reference. It needs no downloaded data or saved checkpoints. **It demonstrates the API; it does not reproduce a paper table or guarantee an improvement.**

```python
from models_hefs import HierarchicalLightGBMRegressor

# X columns must already be arranged as [dominant | complementary].
model = HierarchicalLightGBMRegressor(
    feat_split=(4, 4),
    beta=0.1,
    random_state=42,
    lgb_params_layer1={"n_estimators": 100, "verbosity": -1},
    lgb_params_layer2={"n_estimators": 100, "verbosity": -1},
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Pass numeric arrays, keep time order, and supply exactly the intended `p + q` columns. The estimator slices by position; additional columns beyond that range are not used by the two-layer model. See the [method notes](method.md) for scaling and correction alignment.

## Original experiments

The original scripts are retained with their existing filenames and configurations:

| File | Role |
| --- | --- |
| `den_synt0.py` | Synthetic experiments and hyperparameter search |
| `den_m4n.py` | M4 experiments and hyperparameter search |
| `den_weather.py` | Weather experiments and hyperparameter search |
| `hus_helper.py` | Data construction, evaluation, plotting, statistical comparisons |
| `load_weather2.py` | Weather loading and feature construction |
| `models_hefs.py` | Two-layer HEFS |
| `models_hefs3.py` | Three-layer HEFS |
| `models_ablation.py` | Additive residual ablation |
| `models_baseline.py` | Full-feature LightGBM reference |
| `models_filter.py`, `models_embedded.py` | Feature-selection baselines |
| `models_ensemble.py` | Feature-group ensemble baseline |

### Experiment setup

Each experiment script defines the dataset loader, sample count, feature-group sizes, compared models, and random seeds near the top of the file. The search sections configure `RandomizedSearchCV` with `TimeSeriesSplit` and MAE scoring. Evaluation compares predictions across seeds and summarizes errors with the reporting helpers in `hus_helper.py`.

The experiment utilities use `mlforecast`, `datasetsforecast`, `utilsforecast`, `statsmodels`, `matplotlib`, `joblib`, `tqdm`, and `tqdm_joblib` in addition to the core estimator dependencies. Configure data and output paths for your environment, and use Section 4 and Table 1 of the paper as the reference for its experimental protocol and search ranges.

### Data sources

The paper uses Synthetic (36 features, 300 samples), M4 Yearly (33 features, 400 samples), and Weather (40 features, 300 samples), as listed in Table 2.

- **Synthetic:** the paper describes an ARMA-based signal combined with additional features. Related generators are in `hus_helper.py`; Section 4.2.1 describes the signal construction.
- **M4:** obtain data from the [M4 competition repository](https://github.com/Mcompetitions/M4-methods). The paper uses the Yearly collection; `hus_helper.py` contains loaders for working with M4 data.
- **Weather:** `load_weather2.py` uses Nixtla’s `datasetsforecast.long_horizon.LongHorizon` Weather loader and constructs lag/rolling features. The temperature target is `T (degC)`.

Respect the terms of the original datasets.

## Citation and use

[Read the published paper](https://doi.org/10.1016/j.dsp.2025.105528) and cite it using the repository’s `CITATION.bib`. No code license is currently included in this repository; this documentation does not assign one. The publisher’s PDF and third-party datasets have their own rights and terms.

