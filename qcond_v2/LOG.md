# QV log (host A: embeddings, ZeroMLP, CPU work; node A2: Qwen rewriting). Times local, 2026-11-19/20.

## QV-0
- Read qcond/REPORT_QC.md, REPORT_QC5.md, LOG.md, DESIGN_QC_v1.md and the qcond scripts.
- `DESIGN_QV_v1.md` = section "Frozen design" copied verbatim, sha256 20d165bfc40d1ce1651292605b8a13cd71a941259afc91e9c799a65b8e0b7cd6.
- Reused v1 assets (read through qcond/scripts, not refit): qc3.json (139 practice ids in answerer order, 33 held-out ids, C = 0.01), ~/langres_state/qc/qc3.npz (theta 139 × 4097, stored scores qc, lang, astra, qwen, ref, tl_*), soft_practice.npy (300 × 120), phys_{practice,test}.npy, emb_all.npy, questions.json, practice.json. No Astra or Codex calls in this run; the Codex skill stays in ~/.codex/skills_off/.
- Code: qv_common.py (imports qc_common, qc3 and qc4 unchanged), qv1.py, qv2.py, qv3.py.

## QV-1
- `qv1.py prompts`: 174 prompts, one per question in qcond/questions.json. That is the 139 fitted practice questions, the 33 held-out ones, and the 2 practice questions v1 skipped (A113, P1_5_bottom). P1_5_bottom is needed to check all 31 physical signs; neither enters any fit.
- Node A2 (idle, no GPU jobs): `setsid nohup nice -n 10 python scripts/qv1.py rewrite`. Qwen3.5-9B c2022362…, vLLM 0.34.0, bf16, text only, greedy, enable_thinking=False, guided JSON, max 800 tokens (retry at 1,600 on schema failure), gpu_memory_utilization 0.6, seed 0, page cache evicted. 174 outputs in 36 s, 0 retries, 0 invalid. Per-prompt chat sha256 in rw_out.jsonl. The python process exited on its own (vLLM shutdown message), so no PID was needed to stop it. Raw: ~/langres_state/qv/rw_out.jsonl and rewrite.log (copied from A2).
- `qv1.py check`: physical sign agreement 31/31 = 1.00, which passes. Semantic questions with sign −1: A19, A31, A33, A49, A59, A63, A65, A86 (all practice; no held-out semantic question). -> rewrites.json, rewrite_check.json.
- `qv1.py embed` (host A GPU, nice 10, page cache evicted): Qwen3-Embedding-4B 5cf2132…, bf16, no prefix, normalized, positive forms and negations for all 174 -> ~/langres_state/qv/emb_rewrites.npz.

## QV-2
- Implementation choices (no design change):
  - B2 uses one alpha for both ridges (weights and bias), chosen by LOQO mean AUC.
  - A negation triple uses u = e⁺ of its source and s = −s, because the negation's positive form is the source's positive form. Its base is −s·w(e⁺)·f + b(e of the negation).
  - P and Q start at PCA directions with the mean folded into the bias. With fewer practice questions than 256, Q has at most n − 1 informative PCA directions; the rest are filled with random orthonormal directions (seed = MLP seed). The Q rank is logged.
  - In each fold the base ridge is refit on the training questions only, so the validation questions are scored out of sample.
  - The temperature for A1 is fit by LOQO, as in v1 (A1B1 must reproduce v1). For A2 it uses the 5-fold out-of-fold logits at the chosen epoch, averaged over seeds.
  - Seed scores are averaged on the logit scale before the temperature is applied.
- Smoke run `qv2.py fit 35` (host A GPU, nice 10): 54 s. A2B1 epochs 3/3/3, A2B2 17/15/15.
- `setsid nohup nice -n 10 python scripts/qv2.py fit 139 70`, python PID 2375504 (~/langres_state/qv/fit.pid).
- First full run (139, 70) finished. Its replication check passed on mean AUC (0.6721 both), but T was 1.877 against v1's 1.864, and the per-question LOQO AUC differed from v1 by up to 0.28.
- **QV-E1.** `practice_targets` called v1's `qc3.practice_labels` with the 139 fitted questions only. That function finds a semantic question's soft-label column by its position within the list it is given. v1 passed all 141 practice questions, so with A113 left out, A114–A120 read the column of their neighbour. Fix: call it with all 141 as v1 did and select the 139 columns. Binary labels now agree with the soft targets on 100% of entries. All outputs of the first run (fit_139, fit_70, fit_35 smoke) are discarded and refit.
- Rerun: `setsid nohup nice -n 10 python scripts/qv2.py fit 139 70 35`, python PID 2441671 (the first value recorded was a wrapper and was corrected).
- Rerun done (139, 70, 35 in one process, ~4 min). Replication check: A1B1 mean semantic AUC 0.6721 = v1 (diff 0.0); max probability difference vs v1 2.1e-15; T 1.864 = v1. Pass. replication_check.json.
  - 139: alpha 0.316 for B1 and B2. A1B1 T 1.864, A1B2 T 2.109. A2B1 epochs 5/3/3, T 1.301, OOF AUC 0.791. A2B2 epochs 11/12/8, T 1.877, OOF AUC 0.780. Q rank 138 (B1), 132 (B2).
  - 70: alpha 1.0 (B1), 0.316 (B2). 35: alpha 1.0 (B1), 0.316 (B2). Details in fit_{35,70,139}.json, scores in ~/langres_state/qv/fit_*.npz.

## QV-3
- `qv3.py` (CPU, nice 10, 1,000 paired bootstrap resamples) -> qv3.json, figs/learning_curve.png, figs/cascade_per_answer.png.
- Primary A2B2: mean semantic AUC 0.685 [0.672, 0.697]. R1 +0.044 [0.029, 0.060], passes. R2 ratio 0.831, fails. Cost per answer fails for all four conditions (A2B2 0.622 at 30% of answers; Astra direct 0.731). More data: 139 − 70 = +0.031 [0.024, 0.037], so new questions would likely pay off.
- Effects: polarity −0.008 [−0.017, +0.001]; MLP −0.006 [−0.009, −0.002]; both +0.013 [+0.005, +0.021]; MLP given polarity +0.021 [+0.017, +0.024].
- Held-out questions below 0.5: A1B1 7, A1B2 9, A2B1 7, A2B2 7. Physical mean: 0.602, 0.570, 0.627, 0.583.

## QV-4
- REPORT_QV.md written. No processes left running on host A or A2. The Codex skill remains in ~/.codex/skills_off/. Stopped per design.
