## Frozen design

### Sets (reused)
| Set | Size | Source | Use |
|---|---|---|---|
| Practice | 300 | Random draw (seed 0) from the AH bank of 497 | Practice answers and all fitting |
| Test | 997 | VS Test | Grading |
| Test200 | 199 | Stage 1 Test200 inside Test | Comparison with Astra direct and the cascade |

**Features.**
- **F:** Prithvi-EO-2.0-300M blocks 6, 12, 18 and 24, each mean-pooled, then concatenated (4,096-d), and standardized with Practice statistics. Reuse the AH bank features and the VS Test features.
- **D:** the ten VS physical descriptors, reused from AH.

### Questions
Every question has a text and a one-sentence definition.

**Held-out semantic (23, graded with true labels).**
- The 15 VS graded classes (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19).
  - **Wording:** "Does any part of the patch belong to {class name}, following the CORINE Land Cover nomenclature?"
  - **Definition:** the stage 1 Astra definition.
- The 8 stage 1 grouped questions G1 to G8, with their stage 1 texts and keys.
  - **Definitions:**
    - G1, G2, G3 and G4: the union of their member classes.
    - G5 and G6: "as defined by their member classes".
    - G7 and G8: the stage 1 key rule, written in words.

**Physical (31, keys computed by code).**
- **Percentiles.** Fit-free, from the 497-patch bank.
- **Ten descriptors, with plain names:**

  | Descriptor | Plain name |
  |---|---|
  | brightness | brightness |
  | NDVI mean | greenness (NDVI) |
  | NDVI standard deviation | variation in greenness |
  | vegetated share | vegetated share |
  | NDWI mean | water index (NDWI) |
  | water share | water share |
  | NDMI mean | moisture (NDMI) |
  | B11 mean | shortwave infrared 1 brightness |
  | B12 mean | shortwave infrared 2 brightness |
  | B08 standard deviation | near-infrared texture |

- **P1, level (20).**
  - For each descriptor: "Is the patch's {name} in the top quarter of patches?" and "Is the patch's {name} in the bottom quarter of patches?"
  - Key: above the 75th or below the 25th percentile.
- **P2, contrast (8).**
  - "Is the left half {greener / higher in water index / moister / brighter} than the right half?"
  - The same four for top against bottom.
  - The measures are NDVI, NDWI, NDMI and brightness, each as a half mean.
- **P3, residual (3).**
  - Q17 as defined in FB: "Is the patch moister than its visible and near-infrared appearance predicts?"
  - The same construction for B11 and B12 means: OLS on the RL pool against the B02, B03, B04 and B08 means, NDVI mean and NDWI mean. Key: residual > 0.
  - Wording: "Is the patch brighter in shortwave infrared {1 or 2} than its visible and near-infrared appearance predicts?"
- **Definitions.** Each states its formula and threshold in one sentence.
- **Held out (10):**
  - P1 for moisture, shortwave infrared 2 brightness and near-infrared texture (6),
  - P2 for moisture (2),
  - P3 for Q17 and shortwave infrared 2 (2).
- **Practice (21):** the rest.

**Practice semantic (120, written by Astra, 1 call).**
- **Call settings.** The stage 1 image-QA settings, with no images, a fresh empty folder, a read-only sandbox and no tools. Void any call that runs a tool.
- **Prompt (verbatim).**

  > Write 150 yes or no questions that a scientist might ask about a single Sentinel-2 L2A patch, 1.2 km on a side, seen in a true color image (B04, B03, B02) and a false color image (B08, B04, B03). Cover land use and land cover, spatial layout, vegetation condition and season, human activity, water and wetness, and combinations of these. Each question must be answerable by looking at the two images, and roughly 10 to 60 percent of typical European patches should get a yes. Do not ask about location, date or anything outside the images. Do not simply restate a standard land-cover class name as a question. Give each question a one-sentence definition that settles borderline cases.

- **Schema.** `questions` lists entries with `id`, `question` and `definition`.
- **Leakage filter.**
  - Embed every question as "{question} {definition}".
  - Drop any generated question with cosine above 0.85 to any held-out question.
  - Keep the first 120 survivors in Astra's order.
  - If fewer than 120 survive, keep all of them and log the count.

**Embedding.** Qwen3-Embedding-4B at the VS revision, with no instruction prefix. Each question is embedded as "{question} {definition}" and L2-normalized.

### Practice answers
**Astra (120 calls).**
- **Batches.** 5 Practice patches per call, with images and settings exactly as in stage 1. That is 10 images: true color, then false color for each patch.
- **Questions.** Either the first or the second 60 semantic questions, each with its definition.
- **Calls.** 60 patch groups × 2 question halves.
- **Prompt (verbatim).**

  > You see five Sentinel-2 L2A patches, P1 to P5. Each is 1.2 km on a side with north up and has two images in order: first true color (B04, B03, B02), then false color (B08, B04, B03), where near infrared appears red. Images 1 and 2 show P1, images 3 and 4 show P2, and so on. Answer from the images only. Do not run commands or use tools. For each patch and each question below, answer yes or no with a confidence of low, medium or high, using the definition to settle borderline cases. Judge each patch on its own. {questions}

- **Schema (strict).** `patches` lists 5 entries, each with `pid` and 60 answers of `id`, `answer` and `confidence`.
- **Order check, first.**
  - For patch group 1, run both halves twice: once in the given order, and once in reversed question order. That is 4 calls, of which 2 count as main calls.
  - If agreement between the two orders falls below 0.80, stop and report.
- **Soft labels.** yes high 0.97, yes medium 0.80, yes low 0.57, no low 0.28, no medium 0.10, no high 0.015.

**Code.** Exact keys for the 31 physical questions on Practice and Test.

### Training
**1. Answerers.**
- One logistic model per practice question on F over the 300 Practice patches.
- For soft labels, the label is P ≥ 0.5 and the sample weight is |2P − 1|.
- Pick one C for all answerers from {0.01, 0.1, 1}, by mean 5-fold CV log loss on practice labels across questions.
- Skip a question with fewer than 10 of either answer, and log it.

**2. Formula.**
- Ridge from question embedding to answerer parameters (4,096 weights plus bias).
- Choose alpha from `np.logspace(-3, 3, 13)` by leave-one-question-out over practice questions. The score is the mean AUC of the predicted answerer on Practice patches against that question's practice labels.

**3. Temperature.** One scalar T on the predicted logits, fitted by leave-one-question-out log loss on the practice labels.

### Testing (no labels, no Astra)
For each held-out question:
1. Embed the question.
2. Apply the formula to get its answerer.
3. Score each Test patch as sigmoid((w · f + b) / T).

### Comparisons on held-out questions (Test, AUC)
- **Language route (label-free, no Prithvi).** The mean over the 7 VS field embeddings of each Test patch of the cosine to the question embedding.
- **Astra direct.**
  - The stage 1 Test200 answers for the 15 classes. G1 to G8 are derived from them by the stage 1 key rules.
  - Scored as follows: yes high 0.9, yes medium 0.75, yes low 0.6, no low 0.4, no medium 0.25, no high 0.1.
- **Qwen direct.** The stage 1 Test answers, derived and scored the same way.
- **Per-question reference heads on F** (the "one head per question" route, which needs labels for that very question):
  - classes and G1 to G8: stage 1 Astra bank answers restricted to Practice,
  - physical: exact code labels.
- **True-label heads on F** at n = 30, 100 and 300 on Practice (seeds 0, 1 and 2), for labels equivalent.

### Measures
- **Paired bootstrap 95% CIs** over Test patches, 1,000 resamples.
- **R1, reach over language.** On the 23 held-out semantic questions, mean AUC(QC) − AUC(language route). **Passes** when the lower bound exceeds 0.
- **R2, near the many-heads route.** Mean AUC(QC) ≥ 0.9 × mean AUC(per-question Astra reference head), on the same 23 questions.
- **Cost.**
  - Cascade on Test200 for the 15 classes: QC decides with p ≥ 0.5 when |p − 0.5| ≥ τ, and Astra decides otherwise.
  - Sweep τ from 0 to 0.5 in steps of 0.05, and plot mean balanced accuracy against the share of Astra calls.
  - **Passes** when some τ uses Astra on at most 30% of patches and reaches at least Astra direct's mean balanced accuracy minus 0.01.
- **Report only.**
  - ECE (15 bins) and Brier for QC and for the reference heads.
  - The generalization curve: QC AUC on each held-out question against its maximum cosine to any practice question, with a Spearman correlation.
  - Physical held-out results: QC against the language route and the code-label reference.
  - Leave-one-question-out AUC on practice questions.
  - Labels equivalent.
