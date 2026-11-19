# Method

Only the supplied files were used. The actual bank contains 497 aligned patches. JSONL records were aligned by `row`; output rows follow `test_ids.json`, and columns follow the class IDs in `classes.json`. Classes are independent, so probabilities need not sum to one.

The stronger bank model supplied soft pseudo-labels. Yes answers at high/medium/low confidence became 0.97/0.80/0.57; no answers became 0.015/0.10/0.28. These are conservative modeling assumptions, not empirically established confidence calibrations.

Inputs included all 4,096 Prithvi features, the ten physical descriptors, additional transforms resolving small water fractions, signed confidence-coded weaker-model answers, caption TF-IDF unigrams/bigrams, reported cover proportions, and land-cover keyword counts. Numeric transformations were fitted on the bank; the text vocabulary used the supplied unlabeled captions.

We trained linear and radial-basis kernel ridge regressors on individual and combined evidence sources, extra-trees regressors using 64 bank-fitted feature principal components and other evidence, and class-specific, smoothed weaker-answer calibrators. Five shuffled bank folds evaluated predictions against the stronger model's soft targets. Regularization selection pooled class-specific and overall errors. Related kernel variants were averaged, then nonnegative class-specific ensemble weights were fitted to out-of-fold predictions with shrinkage toward a diverse blend.

All component models were refitted on the complete bank. The ensemble produces float32 probabilities clipped to [0.005, 0.995]. No geographic identity is inferred from water color alone, and no class-exclusivity constraint is imposed.

Validation establishes teacher consistency, not true land-cover accuracy: pseudo-label errors, ambiguous class boundaries, and imperfect confidence calibration remain possible. `predict.py` reproduces the output.
