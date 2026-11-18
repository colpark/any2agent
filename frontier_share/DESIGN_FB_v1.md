## Frozen design

### Reused from the pilot (unchanged)
- the 10 target patches and their renderings,
- the 10 main Astra captions,
- the 16 questions, keys and RL-pool thresholds,
- the Prithvi features (RL-pool standardized),
- the 16 Prithvi probes,
- the Phi-4 caption-only answers at w100,
- the Astra image-QA answers.

### New question Q17 (clean shortwave control)
- **Regression.** On the RL pool, fit an ordinary least squares regression of mean NDMI on six inputs:
  - the patch means of B02, B03, B04 and B08,
  - mean NDVI and mean NDWI.
- **Key.** Q17 is yes when the patch's NDMI residual exceeds 0.
- **Text.** "Is the patch wetter than its visible and near-infrared appearance predicts, meaning its NDMI (B8A and B11) exceeds the value expected from its visible and near-infrared statistics?"
- **Probe.** Fit a Prithvi logistic probe for Q17 with the pilot's settings, and log its 5-fold CV accuracy.
- **Report.** Log the regression's R² and the RL-pool yes-rate.
- Astra with images is not asked Q17, which saves calls. By construction, no rendering shows it.

### Bank
- **Patches.** 1,000 RL-pool patches, stratified by label set with seed 0 (largest-remainder rounding, as in `scripts/make_splits.py`).
- **Captions.** Qwen2.5-VL-7B-Instruct at the pinned revision, through vLLM, greedy, max 160 new tokens. Use the any2agent arm B describe prompt and inputs (true color plus false color with its preamble, from `scripts/common.py`). One summary of about 100 words per patch.
- **Features.** The cached, standardized Prithvi features of the bank patches.

### Method 1: neighbors (Phi-4 reader, free)
- **Similarity.** Cosine on the standardized Prithvi features. For each target, take the 5 most similar bank patches.
- **Conditions per target.**
  - **C0.** Caption only. Reuse the pilot's Phi-4 w100 answers. Run Q17 freshly with the same template.
  - **C1.** Caption plus the 5 Prithvi neighbors.
  - **C2.** Caption plus 5 random bank patches (seed 0 per target). This control separates the value of the foundation model's selection from the value of simply seeing other patches.
- **Reader.** Phi-4 at the pinned revision, greedy, the pilot's reader template. Insert this block between the description and the questions:

  > A foundation model that sees six Sentinel-2 bands, including shortwave infrared, which the descriptions never saw, found these five patches {most similar to this one | chosen at random from the dataset}. Their descriptions come from another model: {rank, cosine to two decimals, description, for each}. Use them as evidence about this patch, especially for properties its own description does not settle.

  - **Questions.** Q1 to Q17.
  - **Target text.** The target's Astra w100 summary.

### Method 2: Astra agent with code (2 Astra calls)
- **Workspace.** A fresh folder `~/langres_state/fb/ws_<call>/` holding only:
  - `bank_captions.jsonl` (bank id and caption),
  - `bank_features.npy` (1,000 × 1024),
  - `target_captions.jsonl` (target id and Astra's full main caption JSON),
  - `target_features.npy` (10 × 1024),
  - `questions.json` (Q1 to Q17 text, no keys, no thresholds beyond those in the text).
- **Calls.**
  - **A1, features.** All five files.
  - **A0, control.** No feature files, only captions and questions.
- **Prompt for A1 (verbatim).**

  > You will answer questions about 10 target Sentinel-2 satellite patches, each 1.2 km on a side. The working folder holds: bank_captions.jsonl, short descriptions of 1,000 other patches written by another vision language model from true color and false color images. bank_features.npy, a 1,000 × 1024 array of Prithvi-EO-2.0 foundation model features for those patches, computed from six Sentinel-2 bands including shortwave infrared (B11, B12), which no description saw. Rows match the order of bank_captions.jsonl. Features are standardized per dimension. target_captions.jsonl, detailed descriptions of the 10 targets written by you from the same kind of renderings. target_features.npy, the targets' features, with rows in the same order. questions.json, 17 yes or no questions. No labels, band values or answer keys exist in this folder, and you must not look for any outside it. Use the descriptions and any analysis of the features you find useful, such as similarity, clustering or relating feature directions to what the bank descriptions say. Python with numpy and scikit-learn is available at {venv python path}. Answer every question for every target with yes or no and a confidence of low, medium or high, and give your best guess when uncertain. Also describe your method in at most 200 words.

- **Prompt for A0.** The same text without the two feature files and without any mention of features, similarity or clustering.
- **Output schema** (strict): `answers` lists 10 targets, each with `target_id` and 17 entries of `id`, `answer` (yes or no) and `confidence`. A `method` field is a string.
- **Settings.**
  - The pilot's call settings: `-m gpt-6-astra`, high effort, `--ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check`, `approval_policy="never"`, `web_search="disabled"`, `--output-schema`, `--json`, `-o`.
  - Read-only sandbox, so commands run but nothing writes.
  - `-C` the workspace.
  - No image flags.
  - Wall-clock limit of 45 minutes per call with `timeout`.
- **Audit.**
  - Parse every command execution in the event stream.
  - Void a call that reads a path outside its workspace, other than the Python interpreter and its libraries, or that touches the network.
  - A voided or invalid call counts against the cap. Retry at most once in total.

### Readouts
1. **Accuracy and skill.** Report accuracy and skill (accuracy minus majority accuracy) per tag for every answerer. The tags are visible, shortwave (Q11 to Q13), texture and Q17. Pool all 17 questions as well. The answerers are:
   - the majority baseline,
   - Phi-4 under C0, C1 and C2,
   - the Astra agents A0 and A1,
   - Astra with images (the existing pilot answers, Q1 to Q16),
   - the Prithvi probe.
2. **Key contrasts, each with counts and a paired sign test across the 170 items.**
   - C1 minus C2: what the foundation model's selection adds.
   - A1 minus A0: what feature access adds for Astra.
   - A1 minus the probe: what remains without labels.
3. **Per-question table** with the 10 targets' keys.
4. **Neighbor check by code.** The share of a target's 5 neighbors that share at least one BigEarthNet label with it. This is computed only for the report, and no reader ever sees it.
5. **Methods.** The Astra agent's method text for A1 and A0, plus the list of commands it ran.
