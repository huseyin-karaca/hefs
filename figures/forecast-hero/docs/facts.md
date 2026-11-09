# Facts — every on-screen string and its source
Paper: Karaca, Tumay, Koc, Kozat, "Hierarchical ensemble-based feature selection for time series prediction", Digital Signal Processing 168 (2026) 105528. Code: `code/models_hefs.py`, `code/models_hefs3.py` (read only, not modified).

Figure size: 120 × 100 mm, both versions (`forecast-hero` light web hero, `forecast-hero-detailed`).

## Feature row
- "Feature importance", "x1", "x2", "xp", "xp+1", "xp+2", "xd", the two ellipses, "Dominant features", "Non-dominant features", "s(1)", "s(2)" (feature groups set as bold vectors, \mathbf{s}, at the user's request; the paper's Algorithm 2 sets them in plain math italic): Algorithm 2 lines 1-7 (relevance scores, sort, split index p = number of dominant features; s(1) = first p features, s(2) = next q). Table 1: split locations (p, q) with p + q = num_feat, so s(2) runs to x_d. Code: `X1 = X[:, :p]`, `X2 = X[:, p:p + q]`. Bar heights illustrative; the ellipsis in each group marks that neither group has a fixed size.
- The former "Data: …" line and the "Observed y" tag are removed (user request); y appears only in the expressions ("Target: y", the cost-optimization equation) and in the plot.

## Models and forward computation
- Expression lines, in the user's notation ŷ = f([x1; x2]), Target: y. Pieces as rendered: "ŷ1 = f1(", "s(1)", "), ", "Target: ", "y" and "α = f2([", "s(2)", "; ŷ1]), ", "Target: ", "α" (+ drawn star).
- "Base model", ŷ1 = f1(s(1)), "LightGBM": §3.1 (i); Algorithm 2 lines 10-12. LightGBM is the base learner in all experiments (§4.1).
- "Second-layer model", α̂ = f2([s(2); ŷ1]) (hat drawn separately): Algorithm 2 line 16 "Concatenate s2 and ŷ(1)", lines 17, 23-24; §3.1.3; code `X2_plus = np.column_stack([X2, y_hat1])`, `model_2.fit(X2_plus, alpha_vec)` / `model_2.predict(X2_plus)`. The bracket [·; ·] is that concatenation.
- Tree glyphs: a boosted-tree ensemble. The base model is drawn with a second, staggered row (5 trees + ellipsis), the second-layer model with 2 trees + ellipsis, at the user's request, to mark the base model as the larger learner. Grounding: the code's search grid lets layer 1 grow larger trees than layer 2 (`param_grid_hefs`: num_leaves 31-254 vs 15-126; n_estimators 200-799 for both). The paper's Table 1 lists one shared LightGBM range for all layers (num leaves 15-512, estimators 400-1500), so the size difference is a code fact, not a paper claim. Tree counts are illustrative.
- "Target: ", "y" and "α" with a drawn star (α⋆): the training target of each model, written after its forward equation. f1 is fitted against the observed y (Algorithm 2 line 11; code `model_1.fit(X1, y_scaled)`); f2 against the constructed α⋆ (line 17; `model_2.fit(X2_plus, alpha_vec)`).
- "Base forecast", "ŷ1": output of f1; drawn once, branching solid to f2 and to the combination, dashed to target construction (code: the same `y_hat1` feeds `_compute_alphas` and `X2_plus`).
- "Predicted", "correction", "α" (α̂), "1 + α" (1 + α̂), "×", "Corrected forecast", "ŷ": Eq. (7), Algorithm 2 line 25, ŷ = (1 + α̂)·ŷ1. α̂ keeps its hat on training data. α̂ is not drawn as clipped: `models_hefs.py` lines 137-139 apply `model_2.predict` unclipped (β acts only on the target search). ŷ is orange, the colour of the corrected-forecast curve.

## Target construction (training only)
- "α" (+ drawn star) above the dashed arrow from the block to the second-layer model: the targets it outputs.
- "Cost-optimization block" (the paper's name, §3.1.2: "a cost-optimization block"), "α = arg min ℓ(y, (1 + a) ŷ1)" (drawn α⋆ = arg min ℓ(y, (1 + a) ŷ1)), "a" (light) or "a ∈ [−β, β]" (detailed), "Training only": §3.1.2, Eq. (4), Algorithm 2 line 14. `a` is a dummy search variable, α⋆ the minimiser. The star marks an optimised target derived from y and ŷ1, not an observed "true correction". The star is drawn as a vector five-point star (the font's asterisk and ⋆ are too small at web size).
- Abstractions (documented here, not in the hero): (1) the code aligns observations and base predictions with a one-step shift (`np.roll(y, 1)` in `_compute_alphas`; `y_prev` in `_best_alpha_vec`); the hero is unindexed and conceptual. (2) The code minimises over a finite grid (300 or 30 points, separate positive/negative halves) on the min-max scaled target, with absolute error in `_best_alpha_vec`/`_compute_alphas`; the hero shows the continuous arg min with a generic loss ℓ. (3) hefs3 sets the first sample's α to 0. No out-of-fold predictions, joint training or additive residual correction are shown.

## Arrow legend
- "Prediction flow" (solid), "Training only" (dashed): solid = forward computation used in training and inference (training predictions ŷ1 come from the same f1); dashed = target construction and supervision, used only when fitting. A second legend row under the plot's curves (user request); the samples carry arrowheads and are grey, unlike the curve samples above them.

## Plot (illustrative toy signal, `figure/parts.js`, no paper data)
- "History", "Forecast": the History | Forecast boundary is the forecast origin, not a training/inference split.
- "Observed", "y", "Base forecast", "ŷ1", "Corrected forecast", "ŷ", "Base-forecast error" (detailed only): curves and shaded gap are illustrative. The corrected curve is computed as (1 + α̂)·ŷ1 with α̂ a smoothed, shrunk copy of the ideal ratio y/ŷ1 − 1 plus a small error, so it recovers about 70 % of each event and is never a perfect fit. Future observations are shown for comparison only, never inputs at inference.

## Literal strings (for the gate)
"Target: ", "α = f2([", "; ŷ1]), ", "ŷ1 = f1(", "), ", "Cost-optimization block", "x1", "x2", "xp", "xp+1", "xp+2", "xd", "s(1)", "s(2)", "ŷ1", "α", "a", "y", "ŷ", "Target", "Observed", "Predicted", "correction", "Base forecast", "Corrected forecast", "1 + α", "ŷ1 = f1(s(1))", "α = f2(s(2), ŷ1)", "α  = ", "arg min", " ℓ(y, (1 + a) ŷ1)", "a ∈ [−β, β]".
