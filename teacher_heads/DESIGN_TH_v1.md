## Frozen design

### Samples (kept small on purpose)

| Set | Size | Source | Use |
|---|---|---|---|
| Bank | 500 | Subset of the FB bank (`~/langres_state/fb/bank.json`), stratified by label set, seed 0 | Unlabeled. All three teachers label it and the heads train on it |
| Test | 1,000 | Frozen test split | Grading the heads, Qwen and code |
| Test300 | 300 | Test, stratified by label set, seed 0 | E3 |
| Test200 | 200 | Test300, stratified by label set, seed 0 | Grading Astra as a teacher |
| Pilot10 | 10 | The frontier share pilot patches | Batching check only |

- **Features.** Prithvi features: the cached ones, standardized with RL-pool statistics.
- **Images.** Arm B renderings from `scripts/common.py`: true color plus false color at 448 pixels. Astra gets 512-pixel PNGs, made the pilot's way.
- **Anonymity.** Astra sees random patch ids only, never filenames, tiles, dates or locations.

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
- **Graded classes.**
  - Grade classes with at least 20 positives in Test, and at least 10 in Test200 for Astra.
  - Train heads for every class that has at least 10 practice labels of each answer.

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

### VLM teachers
Both VLM teachers use the same question text: the 19 classes, numbered, each with the Astra definition.

**Astra (main teacher, 142 calls)**
- **Call settings.** The pilot's image-QA settings:
  - `-m gpt-6-astra`, high effort,
  - `--ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check`,
  - read-only sandbox, approvals never, web search disabled,
  - `--output-schema`, `--json`, `-o`,
  - a fresh empty working folder,
  - images attached with `-i`, in order.
  - Void any call that runs a tool or command, and count it against the cap.
- **Five patches per call.**
  - Shuffle the patch order with seed 0.
  - Patch k gets images 2k − 1 (true color) and 2k (false color), for 10 images per call.
- **Prompt (verbatim).**

  > You see five Sentinel-2 L2A patches, P1 to P5. Each is 1.2 km on a side with north up and has two images in order: first true color (B04, B03, B02), then false color (B08, B04, B03), where near infrared appears red. Images 1 and 2 show P1, images 3 and 4 show P2, and so on. Answer from the images only. Do not run commands or use tools. For each patch and each land cover class below, answer whether any part of the patch belongs to it, following the CORINE Land Cover nomenclature, with yes or no and a confidence of low, medium or high. Judge each patch on its own. {classes}

- **Schema (strict).** `patches` lists 5 entries, each with `pid` (P1 to P5) and `answers`: 19 entries of `id`, `answer` (yes or no) and `confidence`.
- **Calls.**
  - **Batching check, first:** 2 calls on Pilot10.
    - Compare the answers for classes 3, 9, 10, 11, 14 and 7 with the pilot's single-patch image-QA answers Q5 to Q10 (60 answers).
    - If agreement falls below 80%, stop and report.
  - **Bank:** 100 calls.
  - **Test200:** 40 calls.

**Qwen3.5-9B (free teacher)**
- **Model.** Qwen3.5-9B, through vLLM in bf16, greedy, thinking mode off, with JSON-schema guided decoding of the 19 answers. One patch per prompt.
  - Pin the revision at TH-0.
  - If it fails to load or run, take the first that works, in this fixed order: Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B-Instruct.
- **Prompt (verbatim).**

  > You see two renderings of the same Sentinel-2 L2A patch, 1.2 km on each side, north up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears red. For each land cover class below, answer whether any part of the patch belongs to it, following the CORINE Land Cover nomenclature, with yes or no and a confidence of low, medium or high. {classes}

- **Runs.** The bank and Test.

### Practice labels (per class, on the bank)
- **Astra.** Its answer.
- **Qwen.** Its answer.
- **Code.** The rule's yes or no, with abstain left unlabeled.
- **Astra + code (main).**
  - If both agree, keep that answer.
  - If they disagree, leave the patch unlabeled for that class.
  - If the code abstains, take Astra's answer.

### Heads (students)
- **Training.** One head per class and per label source (Astra, Qwen, code, Astra + code), trained on the bank.
- **Classifier.** `LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000)` on the standardized 1,024 Prithvi features. C is chosen on teacher labels only.
- **Label-efficiency reference (for grading).**
  - The same classifier, trained on **true** labels of n random bank patches, with n in 10, 30, 100, 300 and 500, and seeds 0, 1 and 2.
  - For n of 30 or less, use a fixed C of 0.1 instead of cross-validation.
  - If a draw lacks an answer class, predict the constant.

### Reliability without truth (per class, from the bank)
- **Reliability.** The mean agreement rate across the teacher pairs that answered: Astra and Qwen, and Astra and code where the code does not abstain.
- **Tiers.** Cut reliability into tertiles: low, medium, high.

### E3: interface
- **Captions.** Arm B describe captions for Test300 from Qwen3.5-9B, greedy, thinking mode off, max 220 tokens.
- **Readout text (fixed template).**
  - Format: "Prithvi readout (heads trained from teacher labels, no human labels): {class} {p:.2f} ({likely if p > 0.7, possible if 0.3 to 0.7, unlikely if p < 0.3}, reliability {tier}) | …", sorted by p, all 19 classes.
  - It uses the Astra + code heads.
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
**E1, teachers.**
- Measure per class:
  - balanced accuracy (BA) with a 95% bootstrap CI, over 1,000 resamples of patches,
  - for Astra on Test200,
  - for Qwen and code on Test, and also on Test200 for a side-by-side view,
  - the code teacher's coverage (the share it does not abstain).
- Also report the Spearman correlation across classes between reliability and Astra's true BA.
- **Gate:** at least one teacher has a BA CI lower bound above 0.5 on at least 8 graded classes. If not, stop and report.

**E2, heads on Test.**
- Measure per class:
  - BA and AUC for every head,
  - the head's BA minus its own teacher's BA, with a paired bootstrap CI. Astra-taught heads are compared on Test200. Qwen- and code-taught heads are compared on Test.
  - **teacher strength:** Astra + code heads minus Qwen heads, on Test,
  - the label-efficiency curve,
  - the "labels equivalent" of the Astra + code heads: the smallest n whose mean true-label BA reaches them, interpolated in log n, or "more than 500".
- **Gate:** over the classes that passed E1, the Astra + code heads match or beat Astra.
  - Match means a mean BA difference with a CI lower bound above −0.02.
  - Also report whether the Qwen heads beat Qwen (weak to strong).
  - If the gate fails, stop and report.

**E3, reader on Test300.**
- Measure accuracy per question and pooled over the 2,400 items, plus a two-sided McNemar test for R1 against R0 and for R1 against R1s.
- **Gate:** R1 beats R0 by at least 3 points pooled with p < 0.05, and R1 beats R1s.
