## Frozen design

### Samples (kept small on purpose)

| Set | Size | Source | Use |
|---|---|---|---|
| Bank | 1,000 | The FB bank ids (`~/langres_state/fb/bank.json`), RL pool | Unlabeled. Teachers label it and heads train on it |
| Test | 1,000 | Frozen test split | Grading E1 and E2 |
| Test300 | 300 | Test, stratified by label set, seed 0 | E3 |

- **Features.** Prithvi features: the cached ones, standardized with RL-pool statistics.
- **Images.** Arm B renderings from `scripts/common.py`: true color plus false color at 448 pixels.

### Classes
- **The 19 classes.** The 19 BigEarthNet classes, numbered 1 to 19 in the standard order:
  1. Urban fabric
  2. Industrial or commercial units
  3. Arable land
  4. Permanent crops
  5. Pastures
  6. Complex cultivation patterns
  7. Land principally occupied by agriculture, with significant areas of natural vegetation
  8. Agro-forestry areas
  9. Broad-leaved forest
  10. Coniferous forest
  11. Mixed forest
  12. Natural grassland and sparsely vegetated areas
  13. Moors, heathland and sclerophyllous vegetation
  14. Transitional woodland, shrub
  15. Beaches, dunes, sands
  16. Inland wetlands
  17. Coastal wetlands
  18. Inland waters
  19. Marine waters
- **Graded classes.** Grade E1 and E2 on classes with at least 20 positives in Test. Train heads for every class that has at least 10 practice labels of each answer.

### Code teacher (1 Astra call)
- **Workspace.** A fresh folder holding only:
  - `sample_bands.npy`: 50 unlabeled RL-pool patches outside the bank, shape 50 × 12 × 120 × 120, float32, scaled as in `scripts/build_arrays.py`,
  - `classes.json`: number and name.
- **Call settings.** As in the FB agent calls: `-m gpt-6-astra`, high effort, read-only sandbox, approvals never, web search disabled, `--output-schema`, `--json`, `-o`, the venv Python, a 45-minute limit, and the same audit.
- **Prompt (verbatim, fill the scaling sentence from `build_arrays.py`).**

  > Write Python code that labels Sentinel-2 L2A patches with the 19 CORINE land cover classes in classes.json from band values alone, using your knowledge of remote sensing such as spectral indices, texture and simple spatial statistics. No labels exist. sample_bands.npy holds 50 unlabeled patches, shape 50 × 12 × 120 × 120, float32, {scaling}, band order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12, all resampled to 10 m, so each patch covers 1.2 km. Use it only to check that your code runs and gives plausible class rates. Define label(x) taking one 12 × 120 × 120 array and returning a dict from class number to "yes", "no" or "abstain". Abstain when the bands cannot settle a class. Use only numpy and scipy, with no file, network or system access. Also write a one-sentence definition of each class for a vision-language model that sees true color and false color images.

- **Schema.**
  - `definitions`: 19 entries, each with `id`, `name` and `definition`.
  - `code`: a string.
  - `notes`: a string of at most 150 words.
- **Check before running the code.**
  - The imports must be only numpy, scipy and math.
  - The code must contain no `open`, `os`, `sys`, `subprocess`, `socket`, `eval`, `exec` or `__import__`.
  - Run it in a subprocess.
  - A failed check counts as a failed call.

### VLM teacher (free)
- **Model.** Qwen3.5-9B (March 2026, natively multimodal, Apache 2.0), through vLLM in bf16, greedy, thinking mode off, with JSON-schema guided decoding: 19 answers, each yes or no with a confidence of low, medium or high.
  - Pin the revision at TH-0 and confirm that vLLM loads it with two images.
  - If it fails to load or run, take the first that works, in this fixed order: Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B-Instruct (pinned). Log the choice and never switch for any reason other than a load or run failure.
  - Use the same model for every VLM step in this stage, the E3 captions included.
- **Prompt (verbatim).**

  > You see two renderings of the same Sentinel-2 L2A patch, 1.2 km on each side, north up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears red. For each land cover class below, answer whether any part of the patch belongs to it, following the CORINE Land Cover nomenclature, with yes or no and a confidence of low, medium or high. {the 19 classes, numbered, each with the Astra definition}

- **Runs.**
  - Bank, pass 1 (classes in order).
  - Bank, pass 2 (classes in reverse order), used only for self-consistency.
  - Test, pass 1.

### Practice labels
- **VLM.** The pass 1 answer.
- **Code.** The rule's yes or no, with abstain left unlabeled.
- **Both.**
  - If the two teachers agree, keep that answer.
  - If they disagree, leave the patch unlabeled for that class.
  - If the code abstains, take the VLM answer.

### Heads (students)
- **Training.** One head per class and per label source (VLM, code, both), trained on the bank.
- **Classifier.** `LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000)` on the standardized 1,024 Prithvi features. C is chosen on teacher labels only.
- **Label-efficiency reference (for grading).**
  - The same classifier, trained on **true** labels of n random bank patches, with n in 10, 30, 100, 300 and 1,000, and seeds 0, 1 and 2.
  - For n of 30 or less, use a fixed C of 0.1 instead of cross-validation.
  - If a draw lacks an answer class, predict the constant.

### Reliability without truth (per class, from the bank)
- **Self-consistency.** VLM pass 1 against pass 2 agreement.
- **Teacher agreement.** VLM and code agreement on the patches where the code does not abstain.
- **Reliability.** The mean of the two. Cut it into tertiles: low, medium, high.

### E3: interface
- **Captions.** Arm B describe captions for Test300 from the VLM teacher model, greedy, thinking mode off, max 220 tokens.
- **Readout text (fixed template).**
  - Format: "Prithvi readout (heads trained from teacher labels, no human labels): {class} {p:.2f} ({likely if p > 0.7, possible if 0.3 to 0.7, unlikely if p < 0.3}, reliability {tier}) | …", sorted by p, all 19 classes.
  - It uses the "both" heads.
- **Reader.** `microsoft/phi-4` at the pinned revision through vLLM, greedy, with guided JSON of 8 yes/no answers. Use the frontier share reader template. The readout block goes between the caption and the questions.
- **Eight grouped questions.** Their keys come from the true labels for grading only. No head answers any of them directly.

| Id | Question | Yes if the labels include |
|---|---|---|
| G1 | Is any forest present? | 9, 10 or 11 |
| G2 | Is any farmland present? | any of 3 to 8 |
| G3 | Is any built-up land present? | 1 or 2 |
| G4 | Is any open water present? | 18 or 19 |
| G5 | Is any wetland present? | 16 or 17 |
| G6 | Is any natural open land present, such as grassland, heath, shrub or sand? | any of 12 to 15 |
| G7 | Does the patch mix farmland with forest, wetland or natural open land? | G2 and (G1 or G5 or G6) |
| G8 | Does the patch contain more than three land cover types? | more than 3 labels |

- **Conditions on Test300.**
  - **R0.** Caption only.
  - **R1.** Caption plus the patch's own readout.
  - **R1s.** Caption plus another patch's readout (a fixed derangement, seed 0). This one control separates readout content from the priming effect of listing class names.

### Measures and gates
**E1, teachers on Test.**
- Measure per class:
  - balanced accuracy (BA) with a 95% bootstrap CI, over 1,000 resamples of patches,
  - the code teacher's coverage (the share it does not abstain).
- Also report the Spearman correlation across classes between reliability and the VLM's true BA.
- **Gate:** at least one teacher has a BA CI lower bound above 0.5 on at least 8 graded classes. If not, stop and report.

**E2, heads on Test.**
- Measure per class:
  - BA and AUC for each head,
  - the head's BA minus its own teacher's BA, with a paired bootstrap CI,
  - the label-efficiency curve,
  - the "labels equivalent" of the best head: the smallest n whose mean true-label BA reaches it, interpolated in log n, or "more than 1,000".
- **Gate:** over the classes that passed E1, the heads beat their teachers on average with a CI excluding 0, on at least 60% of those classes. The "both" heads must match or beat the best single-teacher heads on average. If not, stop and report.

**E3, reader on Test300.**
- Measure accuracy per question and pooled over the 2,400 items, plus a two-sided McNemar test for R1 against R0 and for R1 against R1s.
- **Gate:** R1 beats R0 by at least 3 points pooled with p < 0.05, and R1 beats R1s.
