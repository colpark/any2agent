## Frozen design

### Sets (reused, no new sampling)
| Set | Size | Source | Use |
|---|---|---|---|
| Bank | 500 | Stage 1 bank (`teacher_heads/sets.json`, `bank_ids`) | Unlabeled. All sources label it and all heads train on it |
| Test | 997 | VS Test (stage 1 Test minus the 3 patches VS dropped) | Grading |
| Test200 | at most 200 | Stage 1 Test200 that falls inside Test | Grading against Astra direct |

### Classes
- The 15 VS graded classes, numbered as in BigEarthNet: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19.
- Each class carries the stage 1 Astra definition (`teacher_heads/code_teacher.json`).
- **H1 classes:** classes with at least 10 positives in Test200.

### Inputs per patch
- **F.** Prithvi-EO-2.0-300M blocks 6, 12, 18 and 24, each mean-pooled, then concatenated (4,096-d). Use the VS extraction code, standardized with bank statistics. Extract the bank fresh, and reuse VS for Test.
- **VC.** The VS text substrate: 7 fields embedded with Qwen3-Embedding-4B and concatenated.
  - Test: reuse the VS embeddings.
  - Bank: run the VS caption step (VS-X1 budget: 3,000 tokens, retry at 4,500), the code field and the embedding step on the 500 bank patches, unchanged.
  - Drop a patch whose caption fails twice, and log it.
- **D.** The ten VS physical descriptors as numbers (brightness, NDVI mean and standard deviation, vegetated share, NDWI mean, water share, NDMI mean, SWIR1 mean, SWIR2 mean, NIR texture), standardized with bank statistics.

### Sources (labeling functions on the bank)
1. **Astra VLM (reuse).** Stage 1 Astra answers on the bank: yes or no, with confidence.
2. **Qwen VLM (reuse).** Stage 1 Qwen3.5-9B answers on the bank and on Test.
3. **Code (1 new Astra call).** Astra writes a physics rule per class.
   - **Workspace** (fresh, read-only sandbox):
     - `bank_descriptors.npy` (500 × 10),
     - `descriptors.json` (names and formulas),
     - `sample_bands.npy` (the first 50 bank patches, 50 × 12 × 120 × 120, float32, scaled as in `scripts/build_arrays.py`),
     - `classes.json` (the 15 classes with definitions).
   - **Call settings.** The stage 1 code-teacher settings, the venv Python on PATH (TH-E1), a 45-minute limit and the same audit.
   - **Prompt (verbatim, with {scaling} filled from `build_arrays.py`).**

     > Write Python code that labels Sentinel-2 L2A patches with the 15 CORINE Land Cover classes in classes.json, using physical reasoning about band values only. No labels exist. bank_descriptors.npy holds ten band statistics for 500 unlabeled patches, with names and formulas in descriptors.json. sample_bands.npy holds 50 of those patches as 12 × 120 × 120 arrays, {scaling}, band order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12, all at 10 m, so each patch covers 1.2 km. Use them to check that your code runs and to set thresholds from the distribution. Define label(desc, x), taking a dict of the ten descriptors and one 12 × 120 × 120 array, and returning a dict from class number to "yes", "no" or "abstain". Commit to yes or no whenever the physics gives real evidence, and abstain otherwise. Use only numpy and scipy, with no file, network or system access. For each class, also state the evidence your rule uses in at most 15 words.

   - **Schema.** `code` is a string. `evidence` lists 15 entries, each with `id` and `text`.
   - **Static check.** The stage 1 static check (imports only numpy, scipy and math, no forbidden names), then run it in a subprocess on the bank and on Test. A failed check counts as a failed call. Retry once within the cap.
   - Log its coverage, the share of non-abstain answers, per class.

### Fusion (truth-free)
**Dawid-Skene, per class.**
- **Parameters.** One sensitivity and one specificity per source, plus a class prior. Smooth each rate with Beta(2, 2).
- **Fitting.**
  - Start EM from the majority vote.
  - Skip abstains.
  - Stop after 200 iterations or when no parameter changes by more than 1e-6.
- **Output.** P(yes) for every bank patch.
- **Fallback.** If EM sets a class's prior below 0.01 or above 0.99, use the plain majority vote for that class, and log it.
- **Inputs.** Astra and Qwen answers count as hard yes or no. Code answers count as yes, no or abstain.

### Heads
- **Classifier.** The stage 1 classifier: `LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000)` on standardized inputs, trained on the bank.
- **Fused labels.** The label is P ≥ 0.5, and the sample weight is |2P − 1|.
- **Single-source labels.** Hard labels with abstains dropped. Train a head only when it has at least 10 of each answer.
- **Heads to train.**
  - **Fused:** on F, VC and D.
  - **Single source:** Astra, Qwen and code labels, on F.
  - **Reference:** true bank labels on F at n of 30, 100 and 500 (seeds 0, 1 and 2, with a fixed C of 0.1 at n = 30), for grading only.

### Direct priors on Test (no head)
- **Astra direct:** the stage 1 Test200 answers.
- **Qwen direct:** the stage 1 Test answers.
- **Code direct:** the code rules applied to Test, with abstain scored 0.5.
- **Scoring.** Score each answer as follows: yes high 0.9, yes medium 0.75, yes low 0.6, no low 0.4, no medium 0.25, no high 0.1.

### Free agent (1 new Astra call)
- **Workspace.** A fresh folder, with the venv Python on PATH, holding only:
  - `bank_features.npy`, `test_features.npy` (F),
  - `bank_descriptors.npy`, `test_descriptors.npy`, `descriptors.json` (D),
  - `bank_captions.jsonl`, `test_captions.jsonl` (the VS caption JSON),
  - `bank_qwen.jsonl`, `test_qwen.jsonl`,
  - `bank_astra.jsonl`,
  - `classes.json`,
  - `test_ids.json`.
- **Call settings.** As for the code call, but with the sandbox set to workspace-write, network off and a 60-minute limit. Apply the same audit, and void any read outside the workspace other than the interpreter and its libraries.
- **Prompt (verbatim).**

  > You will predict land cover for 997 Sentinel-2 test patches, each 1.2 km on a side, without any labels. The working folder holds, for 500 bank patches and the 997 test patches: Prithvi-EO-2.0 foundation model features (4,096 standardized numbers per patch), ten physical band statistics per patch (named in descriptors.json), a structured description of each patch written by a vision language model from true and false color images, and that model's yes or no answers with confidence for each class. For the bank only, bank_astra.jsonl holds answers from a stronger vision language model. classes.json lists the 15 CORINE Land Cover classes with definitions. No true labels exist anywhere, and you must not look for any outside this folder. Use any method you like, such as combining the answer sources, training models on the features with pseudo labels, or physical rules. Python with numpy, scipy and scikit-learn is available. Write scores.npy, shape 997 × 15, each entry your probability that the class is present, with rows in the order of test_ids.json and columns in the order of classes.json. Also write method.md describing your method in at most 300 words.

- **Schema.** `done` is a boolean and `summary` is a string of at most 100 words. Validate `scores.npy` for shape, finite values and the range 0 to 1.

### Measures (AUC throughout, with true labels for grading only)
- **Paired bootstrap 95% CIs** over Test patches, from 1,000 resamples.
- **H1, expansion.**
  - Over the H1 classes on Test200, take the mean of AUC(fused F head) − max(AUC of Astra direct, Qwen direct, code direct).
  - The max is taken per class on Test200. That favors the baseline, so the test is conservative.
  - **Passes** when the CI lower bound exceeds 0.
  - Also report the same on Test against Qwen direct and code direct.
- **H2, amplification.**
  - Over the 15 classes on Test, take the mean of AUC(fused F head) − max over sources of AUC(single-source F head).
  - **Passes** when the CI lower bound exceeds 0.
- **Substrate.**
  - Mean AUC(fused F) − AUC(fused VC), and mean AUC(fused F) − AUC(fused D), on Test.
  - "Prithvi adds as a learner" when both lower bounds exceed 0.
- **Free agent.** Mean AUC(free agent) − AUC(fused F head), on Test.
- **Label value.** The labels equivalent of the fused F head: the smallest n whose mean true-label AUC reaches it, interpolated in log n, or "more than 500".
- **Trust (report only).** Per class, the Spearman correlation between Dawid-Skene's estimated accuracy for each source and its true balanced accuracy on the bank.
- **Cost table.** Astra calls, wall time and GPU minutes per source and per condition.
