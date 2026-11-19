## Frozen design

### Samples
| Set | Size | Source | Use |
|---|---|---|---|
| Discovery | 1,000 | The FB bank (`~/langres_state/fb/bank.json`) | Fit the PCA and pick exemplars |
| Held out | 1,000 | The frozen Test split from stage 1 | Draw the reader's pairs |

- **Features.** The cached Prithvi features, standardized with RL-pool statistics.
- **Images.** One composite per patch:
  - the 512-pixel true color rendering (B04, B03, B02) on the left,
  - the false color rendering (B08, B04, B03) on the right,
  - 1,024 × 512 pixels, made the stage 1 way.
- **Anonymity.** Models see image positions only, never ids, filenames, tiles, dates or locations.

### Directions
- **PCA.** Fit on the 1,000 × 1,024 standardized discovery features: `sklearn.decomposition.PCA(n_components=20, svd_solver="full")`, no whitening.
- **Direction scores.** The 20 components are the directions. A patch's score on direction d is its projection on component d after centering on the discovery mean.
- **Ends.** For each direction, assign "A" to the high end or the low end with a coin flip (seed d). The other end is "B". Log the assignment.

### Naming (Astra, 1 call per direction, 20 calls)
- **Exemplars.** The 6 discovery patches at the A end (highest or lowest score) and the 6 at the B end. Within each group, shuffle the order with seed d.
- **Images.** 12 composites, A1 to A6 first, then B1 to B6.
- **Call settings.** The stage 1 image-QA settings, unchanged:
  - `-m gpt-6-astra`, high effort,
  - `--ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check`,
  - read-only sandbox, approvals never, web search disabled,
  - `--output-schema`, `--json`, `-o`,
  - a fresh empty working folder,
  - images attached with `-i`, in order,
  - at most 3 calls at once.
  - Void any call that runs a tool or command, and count it against the cap.
- **Prompt (verbatim).**

  > You see twelve Sentinel-2 L2A patches, each 1.2 km on a side with north up. Each image shows one patch, with true color (B04, B03, B02) on the left and false color (B08, B04, B03), where near infrared appears red, on the right. A foundation model that also sees shortwave infrared places patches A1 to A6 (images 1 to 6) at one end of one of its internal feature directions and patches B1 to B6 (images 7 to 12) at the other end. Work out what separates group A from group B. Answer from the images only. Do not run commands or use tools. Give a short name for each end, a definition that lets another vision language model judge from the same kind of image which of two new patches lies closer to end A, and up to four visible cues to check, most telling first. Rate how visible the difference is: low, medium or high.

- **Schema (strict).**
  - `name_a` and `name_b`: strings, at most 8 words each.
  - `definition`: a string of at most 80 words.
  - `cues`: 1 to 4 strings.
  - `visible`: low, medium or high.
  - Log the word counts. Don't enforce them.
- **Vocabulary file.** Save all 20 answers with their PC index and end assignment as `fm_names/names.json`.

### Reading (Qwen3.5-9B, free)
- **Model.** Qwen3.5-9B at the revision pinned in stage 1, through vLLM in bf16, greedy, thinking mode off. Guided JSON constrains the output to `{"answer": "1" or "2"}`.
- **Pairs per direction.** 40 pairs from the held-out set.
  - Each pair has one patch from the top 20% of scores on direction d and one from the bottom 20%, both sampled without replacement with seed d.
  - A seeded coin flip per pair (seed 1000 + d) puts the A-end patch at position 1 or 2.
  - Images 1 and 2 are the two composites.
- **Prompt (verbatim).** Join cues with " | ".

  > You see two Sentinel-2 L2A patches, each 1.2 km on a side with north up. Each image shows one patch, with true color (B04, B03, B02) on the left and false color (B08, B04, B03), where near infrared appears red, on the right. Another model describes a contrast between two ends. End A: {name_a}. End B: {name_b}. Definition: {definition} Cues to check: {cues}. Which patch lies closer to end A, image 1 or image 2?

- **Conditions.** Both run on the same 800 pairs.
  - **D1.** Each direction's own description.
  - **D0s.** Another direction's description, under a fixed derangement of the 20 directions (seed 0). This control separates a description that fits its direction from generic image cues. Score it at its better sign per direction, max(acc, 1 − acc). That keeps the control conservative.

### Report-only diagnostic: visible against shortwave
- **Inputs.** For each discovery patch, compute the mean and standard deviation of each of Prithvi's six bands from the cached arrays (`scripts/build_arrays.py`): B02, B03, B04, B8A, B11 and B12.
- **Fits.** For each direction, fit OLS of the score on:
  - the 8 visible and near-infrared statistics (B02, B03, B04, B8A), giving the visible R²,
  - all 12 statistics, giving the full R².
- **SWIR gain.** Full R² minus visible R².
- No model ever sees these values.

### Measures and verdict
1. **Per direction.**
   - D1 accuracy with a Wilson 95% CI (40 pairs).
   - D0s accuracy at its better sign.
   - Astra's names and visible rating.
   - Explained variance ratio, visible R² and SWIR gain.
   - A direction counts as **named** when its D1 Wilson lower bound exceeds 0.5.
2. **Pooled.**
   - The number of named directions.
   - Mean D1 against mean D0s over the 20 directions, with a two-sided Wilcoxon signed-rank test.
   - The share of the top 20 directions' variance that falls in named directions.
3. **Structure.** Spearman correlations across the 20 directions between D1 accuracy and:
   - PC rank,
   - SWIR gain,
   - Astra's visible rating.
4. **Reader bias.** The share of "1" answers per condition.
5. **Verdict N1.** It passes when at least 10 of 20 directions are named and mean D1 beats mean D0s with p < 0.05.
