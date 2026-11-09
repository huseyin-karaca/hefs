# Self-critique (own pass, revision 3)
Faithfulness, checked against Algorithm 2, Table 1 and the code:
- Topology: s1 → f1; s2 and ŷ1 → f2 (written as the concatenation [s(2); ŷ1], Algorithm 2 line 16, code `column_stack([X2, y_hat1])`); ŷ1 is one output that branches (solid to f2 and ×, dashed to the cost-optimization block); α̂ → (1 + α̂) → ×; × → ŷ. Dashed: block → f2, labelled α⋆. Correct.
- Targets: "Target: y" for f1 (Algorithm 2 line 11), "Target: α⋆" for f2 (line 17), written after each forward equation in ink. y has no arrow any more (user request); it is named in both places where it acts.
- Split: x1, x2, ⋯, xp | xp+1, xp+2, ⋯, xd (Table 1: p + q = d).
- Notation: hats on ŷ1, α̂, ŷ; star on α⋆ only; bold vectors for the feature groups (user's notation; the paper uses plain italic, recorded in facts.md). No clipping drawn on α̂.
- Base model drawn larger (5 trees in two rows vs 2 trees): user request; supported by the code's search grid (layer-1 num_leaves 31-254 vs 15-126), not by the paper's Table 1. Recorded in facts.md.

Checks:
- Erase the dashes: removing the dashed arrows and the cost-optimization block leaves s → f1 → ŷ1, ([s2; ŷ1]) → f2 → α̂, ×, ŷ; the "Target: …" annotations remain as text, which is right: they describe how each model was fitted.
- Craft: the bold s sat 1 pt away from the italic "(" (kerned); the second expression line ran within 3 pt of its box edge (both lines now share the largest size that leaves 5.5 pt each side). Feature row pixel-identical to revision 2 (scope diff).

Rubric (1-5): reads at final size 4 (expression lines about 7 pt); takeaway 5; faithful 5; correct 5; links 5; economy 5 (0.52 words/cm², the rest is the equations); craft 5.
Gate warnings kept on purpose: two font families (maths in Cambria); "Target: " flagged as caption-like for its colon (the user's notation); the drawn hat over α̂ counted as art over text; word density.
