## Frozen design

### Data
- **Source.** BigEarthNet v2 (reBEN) Sentinel-2 L2A patches, 12 bands, 120 × 120 pixels at 10 m. Resample the 20 m bands (B05, B06, B07, B8A, B11, B12) to 10 m with bilinear interpolation.
- **Splits.** Freeze ids and sha256 at LR-1, before any model sees data. Stratify every split by label set.

  | Split | Size | Source | Use |
  |---|---|---|---|
  | Calibration | 200 | official train | centering means only |
  | RL pool | 20,000 | official train, disjoint from calibration | Part B |
  | Val | 100 | official val | smoke tests and change requests |
  | Test | 1,000 | official test | all reported results |

- **Composites.**
  - RGB composite: B04, B03, B02, with a fixed 2 to 98 percentile stretch per band computed on the RL pool.
  - False color composite: B08, B04, B03, with the same rule.
  - Both upsampled to 448 × 448 bicubic for the VLM.
- **Descriptors computed by code** for every patch: mean NDVI (B08, B04), mean NDWI (B03, B08), mean NDMI (B8A, B11), and the 19 class label set.

### Primary VLM (frozen)
- Qwen2.5-VL-7B-Instruct, bf16.
- Readout layer ℓ sits at three quarters of the language model depth (layer 21 if the model has 28 layers). Confirm the layer count at LR-0 and record it. Never sweep ℓ to pick it. A layer sweep may appear only as a diagnostic on the calibration split.

### Arms
- **Arm A.** The VLM sees the RGB composite only.
- **Arm B.** The VLM sees the RGB composite plus the false color composite as a second image. Every arm B prompt that shows images starts with this preamble: "The second image is a false color composite of the same patch with near infrared shown as red."

### Prompts (verbatim)
- **Describe** (generation). Images, the arm preamble, then: "Describe this Sentinel-2 satellite image patch for a scientist in at most 100 words. Cover land cover, vegetation, water, built structures, texture and spatial layout."
- **Pass 1** (gives z).
  - User turn: images, the arm preamble, then "What is the key content of this Sentinel-2 patch?"
  - Assistant turn prefilled with "The key content of this patch is".
  - Read the hidden state of the last prefilled token at layer ℓ.
- **Pass 2** (gives ẑ).
  - User turn, no images: "Here is a description of a Sentinel-2 patch: " + w + " What is the key content of this Sentinel-2 patch?"
  - Same assistant prefill, same token, same layer.
- **Rewrite** (search rounds).
  - Images, the arm preamble, then "Here are earlier descriptions of this patch with their scores. Higher is better." followed by the top 4 descriptions, each with its score to three decimals.
  - Then: "Write one new description in at most 100 words that states what the image shows more completely and precisely. Do not mention scores."

### Score and residual (computed, never learned)
- **Centering means.** μ_img is the mean pass 1 state over the calibration split. μ_txt is the mean pass 2 state over the calibration split's zero-shot greedy summaries. Compute both per arm and freeze them at LR-2.
- **Unit vectors.** u_img = (z − μ_img) / ‖z − μ_img‖ and u_txt = (ẑ − μ_txt) / ‖ẑ − μ_txt‖.
- **Score.** s(w) = cos(u_img, u_txt).
- **Residual.** r = u_img − u_txt, stored with w.
- **Gap closure.** g = (s − s_B0) / (1 − s_B0), where s_B0 is the generic summary's score for the same patch.

### Part A: per-instance search (nothing trains)
1. **Round 0.** The frozen VLM samples 32 summaries with the describe prompt, at temperature 1.0 and 160 new tokens at most.
2. **Rounds 1 to 3.** The rewrite prompt takes the current top 4 and samples 8 new summaries at temperature 1.0.
3. **Total.** 56 generations per instance, all scored with s.
4. **Final pick.** The highest scoring summary of at most 100 words.
5. **Logging.** Save every candidate with its score.

### Part B: amortized RL (comparison)
- **Policy.** The primary VLM plus a LoRA (rank 16, alpha 32) on q, k, v and o of the language model only.
- **Scorer.** The base model with the LoRA switched off runs both passes, so the policy can never move z.
- **Reward.** s(w) minus 0.5 if the summary exceeds 100 words.
- **GRPO.** 4 samples per patch, 32 patches per step, LoRA lr 1e-5, clip 0.2, KL to the base model β = 0.04 (adapter off as the reference), 450 steps over the RL pool, one seed per arm.
- **Framework.** TRL GRPO preferred. Log the choice. The framework is not part of the design, but the hyperparameters above are.
- **Evaluation.** RL greedy (one pass) and RL plus search (Part A procedure starting from the RL policy, same budget).

### Check VLM (frozen, evaluation only)
- The first of InternVL3-8B, then Gemma-3-12B-it, that loads and runs.
- It repeats pass 1 and pass 2 with its own layer at three quarters depth and its own calibration means, giving a check score s′ for every final summary.
- It never takes part in search or reward.

### Conditions per arm, all on the same 1,000 test patches
- **B0 generic.** "A satellite image patch."
- **B1 labels.** "Land cover: " plus the patch's label names.
- **B2 zero-shot.** The greedy describe output.
- **A search.**
- **B RL.**
- **B RL plus search.**

### Secondary target (frozen)
- Prithvi-EO-2.0-300M: the mean of patch tokens on B02, B03, B04, B8A, B11, B12, resized to 224, one time step, with the model's own normalization.
- Standardize per dimension with RL pool statistics.

### Readouts (all from code or frozen models)
1. **Primary score.** s and g per arm and condition. Mean, 95% paired bootstrap CI (1,000 resamples), and paired differences.
2. **Check score.** s′ per arm and condition, plus the correlation between s and s′ across patches.
3. **Prithvi decomposition.** Ridge regression to standardized Prithvi z, 5-fold CV on test with fixed folds and alpha chosen inside each fold.
   - From u_img: what the VLM perceives.
   - From u_txt: what the words carry.
   - Report R² per condition.
4. **Probes.** Ridge for mean NDVI, NDWI and NDMI, and one logistic regression per label (macro F1 and mAP). Use the same 5-fold CV on test.
   - Features: (a) u_txt, (b) r, (c) u_img, (d) u_txt and r together.
   - The key number is the gain of (d) over (a).
5. **Code-keyed QA.**
   - Ten yes/no questions per patch:
     - NDVI above 0.5
     - NDVI below 0.2
     - NDMI above the RL pool median
     - NDWI above 0
     - presence of each of the six most frequent RL pool labels
   - Keys come from code. Llama-3.1-8B-Instruct, used nowhere else, answers greedily from w alone.
   - Report accuracy against the per-question majority baseline.
6. **Gaming checks.**
   - Qwen2.5-7B-Instruct (text only) paraphrases every final search and RL summary. Recompute s and s′ and report the drop.
   - Summary perplexity under the base text model.
   - Word count histogram.
   - Share of non-ASCII characters.
   - 20 random summaries per condition printed in the report.
7. **Part B curves.** Reward, KL and summary length per step.

### Predictions (pre-registered, never tuned toward)
- **P1.** Search raises s over zero-shot, and s′ rises too.
- **P2.** In arm A, adding r to u_txt improves at least one descriptor probe. That would show the verbalization gap holds information.
- **P3.** Arm B raises the Prithvi R² from u_img over arm A, mostly on the NDVI and NDMI probes.
- **P4.** RL greedy reaches at least half of the search gain in s over zero-shot, and RL plus search beats search alone.

Mark each prediction supported, not supported or inconclusive, with the numbers.
