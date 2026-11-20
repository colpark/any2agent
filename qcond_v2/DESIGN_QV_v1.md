## Frozen design

### Reused from QC v1, unchanged
- Practice (300 patches), Test (997), Test200 (199) and the standardized Prithvi features F (4,096).
- The 139 practice questions and their practice labels: Astra soft labels and code keys.
- The 139 v1 answerers θ_q = (w_q, b_q) and the v1 C.
- The 33 held-out questions: 23 semantic and 10 physical.
- The Qwen3-Embedding-4B embeddings, at the same revision and with no prefix.
- The v1 comparison scores: language route, Astra direct, Qwen direct, reference heads and true-label heads. Reuse them as stored. Do not refit them.

### Rewriting (factor B2): Qwen3.5-9B, free
- **Model.** Qwen3.5-9B at the stage 1 revision, through vLLM in bf16, text only, greedy, thinking mode off, with guided JSON.
- **Coverage.** One prompt per question, for all 139 practice and 33 held-out questions.
- **Prompt (verbatim).**

  > Rewrite the yes or no question below about a satellite image patch. First give its positive form: the same property, asked in the direction of presence, a high value, or the direction the property names. Give sign 1 if the original already asks in that direction, and sign -1 if it asks about absence, a low value, the bottom of a range, or the reverse direction. Keep the property and its definition, and change only the direction. Then give the natural negation of the original question, with a definition. Question: {question} Definition: {definition}

- **Schema.** `positive_question`, `positive_definition`, `sign` (1 or -1), `negation_question`, `negation_definition`.
- **Check, model-based.**
  - Compare the signs on the 31 physical questions with their known directions: top quarter 1, bottom quarter -1, P2 and P3 1.
  - If agreement falls below 0.90, stop and report.
  - Log every semantic question that gets sign -1.
- **Embeddings.** Embed the positive form "{positive_question} {positive_definition}" and the negation "{negation_question} {negation_definition}" exactly as in v1.

### Conditions (2 × 2)
Notation: for question q, e is the original embedding, e⁺ the positive-form embedding and s the sign.

**B1, questions as written.** The base score is the v1 formula, refit as in v1: one ridge from e to (w, b).

**B2, positive form, sign and negation pairs.**
- **Weights.** A ridge from e⁺ to s · w_q.
- **Bias.** A separate ridge from e to b_q.
- **Negation pairs.** Add each practice question's negation to the bias ridge as (e of the negation, −b_q).
- **Base score.** s · w(e⁺) · f + b(e).
- **Ridge settings.** Pick alpha from `np.logspace(-3, 3, 13)` by leave-one-question-out AUC on practice labels, as in v1. A negation pair always leaves out together with its source question.

**A1, base only.** score = base(f, q).

**A2, base plus ZeroMLP.**
- **Score.** score = base(f, q) + z(f, q), with the base frozen.
- **Network.** z = MLP([P f, Q u, (P f) ⊙ (Q u), s]).
  - u = e in B1 and u = e⁺ in B2.
  - s = 1 in B1.
  - P (4,096 to 256) starts at the top 256 PCA directions of Practice F.
  - Q (2,560 to 256) starts at the top 256 PCA directions of the practice question embeddings.
  - The MLP is 769 → 512 → 256 → 1, with GELU and dropout 0.1.
  - The last layer starts at zero, so training begins exactly at the base score.
- **Training triples.** Every Practice patch with every practice question and its practice label (Astra soft P, or code key 0 or 1).
  - In B2, also add every negation triple with label 1 − P, using its own u and s.
- **Loss.** Binary cross-entropy with soft targets on the logit score.
- **Optimizer.** AdamW, lr 1e-3, weight decay 1e-2, batch 1,024, at most 50 epochs.
- **Seeds.** 0, 1 and 2. Average their scores.
- **Early stopping.**
  - Use 5 folds grouped by question (seed 0). A negation pair stays in its source question's fold.
  - Pick the epoch count with the lowest mean validation loss across folds, with patience 5.
  - Then retrain on all practice questions for that many epochs.
- **Calibration.** For every condition, fit one temperature T by grouped-CV log loss on practice labels.

**The primary condition is A2 × B2,** fixed in advance. The other three are ablations, and A1 × B1 must reproduce v1.

### Learning curve over practice questions
- **Subsets.** Nested subsets of 35, 70 and 139 practice questions, drawn with seed 0 and stratified by semantic against physical. Negation pairs follow their source question.
- **Fitting.** Refit every condition at each size, with the same selection rules.
- **Report.** Mean held-out semantic AUC against size, for each condition.

### Measures (Test, AUC unless stated)
- **Paired bootstrap 95% CIs** over Test patches, 1,000 resamples.
- **Replication check.** A1 × B1 mean semantic AUC within 0.005 of v1's 0.672, or log a QV-E entry and stop.
- **R1.** Mean AUC over the 23 semantic questions minus the language route. Passes when the lower bound exceeds 0.
- **R2.** Mean AUC ≥ 0.9 × the mean of the per-question reference heads (0.824).
- **Effects, on the 23 semantic questions:**
  - polarity: A1B2 − A1B1,
  - MLP: A2B1 − A1B1,
  - both: A2B2 − A1B1,
  - MLP given polarity: A2B2 − A1B2.
- **Polarity check.**
  - The number of held-out questions with AUC below 0.5. v1 had 7.
  - The mean AUC on the 10 physical held-out questions.
- **Cost, per answer.** This is the right model when one new question is asked.
  - For each of the 15 classes on Test200 separately, send an answer to Astra when |p − 0.5| < τ, and keep QC's answer otherwise, deciding yes at p ≥ 0.5.
  - Sweep τ from 0 to 0.5 in steps of 0.05.
  - Report mean balanced accuracy against the share of answers sent to Astra.
  - **Passes** when some τ sends at most 30% of answers and reaches at least Astra direct's mean balanced accuracy (0.731) minus 0.01.
- **Calibration.** ECE (15 bins) and Brier on the semantic and physical held-out questions.
- **Describing a patch.** The QC-5 per-patch question-ranking AUC, for every condition.
- **More data decision.** In the primary condition, a rise of at least 0.01 in mean semantic AUC from 70 to 139 questions means new Astra questions would likely pay off. Report the rise with its CI.
