# Spec — forecast-hero (revision 3)
Structure: pipeline (features → base model → second-layer model → combination → forecast) with a supporting training-only branch, one unified panel, read top to bottom.
Hero: none, a flat pipeline over an illustrative plot; no physical object, so route = flat.
Size: 120 × 100 mm (website hero), both versions.
Versions (one source, `figure/hero.js`): `figure.js` → `forecast-hero` (light, web hero); `figure_detailed.js` → `forecast-hero-detailed` (adds the search bound a ∈ [−β, β] and the base-forecast error band).
Arrangement: the arrow legend is the second row of the bottom legend (user request, revision 3).

Components
- Feature-importance row: sorted bars x₁, x₂, ⋯, x_p | x_{p+1}, x_{p+2}, ⋯, x_d; dashed split at p; brackets "Dominant features" (blue) / "Non-dominant features" (orange). No "Data:" line.
- Base model (blue): "Base model"; one line "ŷ₁ = f₁(𝐬⁽¹⁾),  Target: y" (equation in blue, target in ink); a tree ensemble in two staggered rows (3 + 2 trees and an ellipsis), "LightGBM".
- Second-layer model (orange): "Second-layer model"; one line "α̂ = f₂([𝐬⁽²⁾; ŷ₁]),  Target: α⋆"; one row of 2 trees + ellipsis, "LightGBM". Output "Predicted correction α̂".
- Cost-optimization block (grey, dashed outline, "Training only"): α⋆ = arg min_a ℓ(y, (1 + a) ŷ₁) (detailed: a ∈ [−β, β]).
- Feature groups are bold vectors 𝐬⁽¹⁾, 𝐬⁽²⁾ everywhere (arrow labels and equations); the expression lines share one size, the largest that fits both boxes.
- Combination ×: ŷ₁ (blue) and 1 + α̂ (orange) → "Corrected forecast ŷ" (orange label, orange arrow).
- Plot: History | Forecast (forecast origin), no axis title; illustrative toy signal; the corrected curve recovers about 70 % of each event.
- Legend, two rows on shared columns: "Observed y", "Base forecast ŷ₁", "Corrected forecast ŷ" (detailed: + "Base-forecast error"); below them the diagram arrows, solid "Prediction flow", dashed "Training only"

Relations
- Solid: s⁽¹⁾ → f₁ (top edge); s⁽²⁾ → f₂ (top edge); ŷ₁ (one output, branching) → f₂ (left edge, lower) and → ×; α̂ → (1 + α̂) → ×; × → ŷ.
- Dashed: ŷ₁ → cost-optimization block (bottom, branch of the same output line); block → second-layer model (left edge, labelled α⋆). y is not drawn as an arrow: it is named in "Target: y" and in the block's equation.
Conventions: blue = dominant/base, orange = non-dominant/second layer/corrected forecast, grey dashed = training only (arrows and the cost-optimization block); hats = predicted, star = optimised target (drawn as a vector star), y = observed. Each model's target is written after its forward equation ("Target: …", in ink); the α⋆ arrow enters the second model's side, never the feature (top) edge. Meaning survives greyscale (dash vs solid, text labels).
Negative: no measured performance; α⋆ not drawn as observed; α̂ not drawn as clipped; no out-of-fold, joint training or additive residual; no indexed timestamps.
