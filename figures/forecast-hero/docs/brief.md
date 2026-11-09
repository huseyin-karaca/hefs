# Brief — forecast-hero (revisions 2 and 3, polish)

Mode: revise. Format: journal style, 120 × 100 mm, used as the project-page hero (shown about 720 px wide on desktop).
Reader: a visitor to the paper page who has not read the paper and should get the idea in a few seconds.

Takeaway: A base model predicts from the dominant features; a second model uses the complementary features and that base prediction to predict a relative correction. Training uses observed outcomes to construct correction targets; inference uses the two trained models and combines their outputs.

The user's own criteria (this pass, verbatim in substance):
1. Remove the "Data: ..." line on top unless it is shown beautifully, integrated into the figure.
2. The dominant group must not look fixed at four features: x1, x2, ⋯, xp | xp+1, ⋯, xd.
3. Reduce unimportant details, or give a lighter and a more detailed version.
4. Show the target inputs better.
5. Show the base model as the bigger model with a second line of trees (in the base model only, not a 3 × 2 grid).
6. Find a better place for the arrow legend.
7. "Corrected forecast ŷ" in orange.
8. The corrected forecast must not overlap the observed y so much that it looks like a perfect forecast.

Kept from revision 1: one unified diagram (no training / inference panels, no duplicated models); solid = prediction flow, dashed = training only; hats = predicted, star = optimised target; equations ŷ1 = f1(s(1)), α̂ = f2(s(2), ŷ1), ŷ = (1 + α̂)ŷ1; supervision enters separately from features.

Must not imply: measured performance; α⋆ observed; α̂ clipped; out-of-fold predictions, joint training or additive residuals; same-timestamp alignment.

Revision 3 requests (user, verbatim in substance):
1. Write the model expressions in the notation ŷ = f_θ([x₁; x₂]), \qquad Target: y, and remove the "Target" boxes inside the models.
2. Rename "Compute correction targets" to the cost-optimization block.
3. Two LightGBM trees instead of three in the second-layer model.
4. Remove the "Observed y" block and its arrows.
5. Remove "Time" from the axis.
6. Put the diagram-arrows legend at the bottom, as a second row under Observed y / Base forecast / Corrected forecast.
