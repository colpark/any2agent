## Frozen design

### Samples
| Set | Size | Source | Use |
|---|---|---|---|
| Fit | 2,000 | RL pool, stratified by label set, seed 0 (`scripts/make_splits.py` logic) | Fit maps and probes |
| Test | 1,000 | The frozen Test split from stage 1 | All held-out measures |

### Prithvi features (F)
- **Extraction.** Use the any2agent Prithvi-EO-2.0-300M extraction code unchanged, plus forward hooks on all 24 transformer blocks.
- **Per-layer token.** Mean-pool each block's output over spatial tokens, excluding any CLS token. That gives one 1,024-d token per layer.
- **Standardization.** Standardize each layer with Fit-set statistics.
- **F.** The concatenation of blocks 6, 12, 18 and 24 (4,096-d). The per-layer tokens serve only VS-2's layer curve.

### Descriptions
**VLM caption (Qwen3.5-9B).**
- **Model.** The stage 1 revision, through vLLM in bf16, greedy, thinking mode off, with JSON-schema guided decoding.
- **Inputs.** The stage 1 Qwen inputs: arm B true color and false color renderings.
- **Prompt.** The frontier share pilot caption prompt, verbatim for its opening, its rules and levels 1 to 6. Replace the nested-summary sentence with: "Then write one summary for a scientist of at most 100 words, in plain prose. Return only JSON that matches the schema."
- **Schema.** `frontier_share/schemas/caption.json`, copied to `verbal_share/schemas/caption_vs.json`, with `summaries` replaced by one string field `summary`.
- **Length and retries.**
  - Max 1,000 new tokens.
  - On a schema failure, rerun once with max 1,500.
  - If it still fails, drop the patch and log it.

**Code description (deterministic).**
- **Descriptors.** Ten descriptors from the cached 12-band arrays (`scripts/build_arrays.py` scaling):
  1. brightness, the mean of B02, B03 and B04,
  2. NDVI mean, NDVI = (B08 − B04)/(B08 + B04),
  3. NDVI standard deviation,
  4. vegetated share, the fraction of pixels with NDVI > 0.5,
  5. NDWI mean, NDWI = (B03 − B08)/(B03 + B08),
  6. water share, the fraction of pixels with NDWI > 0,
  7. NDMI mean, NDMI = (B8A − B11)/(B8A + B11),
  8. SWIR1 mean (B11),
  9. SWIR2 mean (B12),
  10. NIR texture, the standard deviation of B08.
- **Sentences.** Each descriptor becomes one sentence: "{Name} is {word} ({p}th percentile of the region), value {v:.2f}."
  - The percentile p comes from the Fit set.
  - The word is very low below 10, low from 10 to 30, typical from 30 to 70, high from 70 to 90 and very high above 90.

**Text fields and embeddings.**
- **Fields.** Turn each caption into six text fields, joining each JSON part into plain sentences:
  1. global,
  2. layout (the nine grid cells plus boundaries),
  3. features,
  4. spectral,
  5. interpretation,
  6. summary.
  - Log the uncertainties, but don't embed them.
  - The code description is field 7.
- **Embedding model.** Qwen3-Embedding-4B at a pinned revision, with no instruction prefix. If it fails, use Qwen3-Embedding-0.6B.
  - Embed each field separately and L2-normalize it.
- **Representations.**
  - **V:** the concatenation of fields 1 to 6 (vision language).
  - **VC:** V plus field 7 (vision plus physical language).

### VS-2: verbal share (no labels)
The geometry between the two spaces may be nonlinear, so the run uses three maps from text (V or VC) to F, all fit on Fit.

**Maps.**
- **Linear.** `RidgeCV(alphas=np.logspace(-1, 5, 13), alpha_per_target=True)`.
- **Kernel.** `KernelRidge(kernel="rbf")`.
  - gamma comes from {0.25, 0.5, 1, 2, 4} divided by the median squared distance on Fit.
  - alpha comes from `np.logspace(-3, 1, 5)`.
  - One pair applies to all targets, chosen by 5-fold CV on Fit.
- **Neighbors.** `KNeighborsRegressor(metric="cosine", weights="distance")`, with k from {5, 10, 20, 50} chosen by 5-fold CV on Fit. This map is fully nonparametric.

**Selection.** Choose the map with the best 5-fold CV R² on Fit. That map defines F̂ and the residual. Report all three on Test. The gap between linear and the best nonlinear map measures how bent the geometry is.

**Verbal share.**
- Held-out R² on Test, weighted by variance across dimensions: 1 − Σ(F − F̂)² / Σ(F − mean_Fit F)².
- Report a bootstrap 95% CI over Test patches (1,000 resamples).
- Treat it as a lower bound, since any map from finite data understates it.

**Geometry check (no map).** Agreement between VC space and each layer, on Test:
- mutual k-nearest-neighbor overlap (k = 10, cosine),
- linear CKA,
- RBF CKA with the median bandwidth,
- each against VC rows shuffled with seed 0.

These depend on no fitted map, so nonlinear bending cannot hide shared structure.

**Curves and tables.** These use the selected map family, with hyperparameters re-chosen by CV on Fit each time, and the linear map alongside.
- **Layer curve.** Each of the 24 per-layer tokens, for V and VC.
- **Field table.** The verbal share of F from each of the 7 fields alone.
- **Learning curve.** VC to F at n of 250, 500, 1,000 and 2,000, using nested Fit subsets with seed 0.

**Sanity check.** VC rows shuffled across Fit patches (seed 0) should give an R² near 0.

**Residual.** R = F − F̂ from the selected VC map, so F = F̂ + R in the same coordinates.
- On Fit, use 5-fold cross-fitted predictions (KFold, seed 0).
- On Test, use the map fit on all of Fit.
- Residual share = 1 − verbal share.

### VS-3: what the residual holds (labels as a measuring instrument)
- **Probes.** The stage 1 classifier, `LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000)`, trained on Fit and scored on Test by AUC. Each input is standardized.
- **Inputs.** V, VC, F, F̂ (the verbal part), R, and VC + F (concatenated).
- **Targets.**
  - The graded classes: those of the 19 BigEarthNet classes with at least 20 positives in both Fit and Test, with true labels.
  - Q17: the FB key, the sign of the NDMI residual after OLS on the B02, B03, B04 and B08 means, mean NDVI and mean NDWI, fit on the RL pool. The descriptions never state this residual.
- **Contrasts.** Each with a paired bootstrap 95% CI over Test patches (1,000 resamples):
  - **FM gain beyond priors:** AUC(VC + F) − AUC(VC), per target and averaged over the graded classes, with Q17 reported separately.
  - **The same gain in F's coordinates:** AUC(F) − AUC(F̂). F̂ carries the text through the nonlinear map, so a linear probe reads F̂, R and F on equal terms.
  - **Physical-language gain:** AUC(VC) − AUC(V), per target.
  - **Residual content:** AUC(R) per target, against 0.5.

### Headline readouts
- **H1.** Verbal share of F from V and from VC, with CIs, from the selected map. Report the linear share and the geometry check beside it. This splits Prithvi into vision language, added physical language and residual.
- **H2.** FM gain beyond priors, as the mean over classes and for Q17, in both forms. "Prithvi adds beyond the priors" when both CI lower bounds exceed 0.
- **H3.** Residual content. "The addition lives in the residual" when AUC(R) has a CI lower bound above 0.5 on the targets where H2 holds.
