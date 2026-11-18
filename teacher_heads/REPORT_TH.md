# Teacher-trained heads, stage 1 (TH-0 to TH-2, stopped at the E2 gate): report

**Gate verdicts.**
- **E1 passes.** Astra beats guessing on all 12 of its graded classes (BA CI lower bound above 0.5), and Qwen3.5-9B on 10 of 15. The code teacher covers no graded class.
- **E2 fails.** On Test200, the Astra + code heads trail Astra by a mean of 0.031 BA (95% CI −0.058 to −0.004). "Match" needed a CI lower bound above −0.02. Two rare classes cause most of the gap. On most classes the heads equal their teacher, and on 7 of 15 classes they beat a head trained on 500 true labels.
- **E3 was not run**, because the design stops at a failed E2 gate.

## Deviations from DESIGN_TH_v1.md

| Id | Deviation | Status |
|---|---|---|
| — | none | |

Bug fixes and notes (LOG.md has details):
- **TH-E1:** the venv Python first on PATH for the agent call, which restores the design's "the venv Python". Code-teacher call 1 was voided for real reads outside its workspace. Call 2 passed.
- **TH-E2:** the audit wrapper drops two false-positive classes: numpy `~`, and "ssh" inside a folder name.
- My smoke-test script needed a `main()` guard. That was a script bug, not a model failure, so there was no model fallback.
- The prompt was updated mid-TH-0. The first design is kept as DESIGN_TH_v1_superseded_20261118.md.
- The readout template covers only classes that have a trained head (E3 not run).

## Setup facts
- **Astra calls:** 144 of 160. That is 2 code-teacher calls (1 voided, 1 ok), 2 batching-check calls and 140 teacher calls. All 140 teacher calls were valid on the first try, with 0 tool uses and 0 usage limits. Each took about 40 s at concurrency 3.
- **Batching check:** 48 of 60 answers (0.80) agree with the pilot's single-patch answers. The check passes, but exactly at the threshold.
- **Qwen3.5-9B:** pinned at revision c2022362…, run with vLLM bf16, thinking off and guided JSON, 0 unparsed outputs. It took 414 s for the bank and 836 s for Test.
- **Code teacher (Astra's own choice):** "an exclusion-only baseline … it deliberately never emits 'yes'". Its coverage is 0.8% on the bank and only for classes 18 and 19. So no code head was trainable, and the Astra + code labels equal Astra's labels.
- **Heads:** 47 trained (Astra, Qwen and Astra + code; none from code). The run took 91 s on CPU.

## E1: teachers (balanced accuracy with 95% CI)

| Class | Test pos | Astra (Test200) | Qwen (Test) | Reliability (tier) |
|---|---|---|---|---|
| 1 Urban fabric | 96 | 0.80 [0.67, 0.91] | 0.79 [0.75, 0.84] | 0.90 med |
| 3 Arable land | 420 | 0.77 [0.71, 0.83] | 0.77 [0.74, 0.80] | 0.87 med |
| 4 Permanent crops | 34 | — (<10 in Test200) | 0.52 [0.49, 0.57] | 0.92 high |
| 5 Pastures | 217 | 0.74 [0.68, 0.79] | 0.66 [0.63, 0.69] | 0.64 low |
| 6 Complex cultivation | 178 | 0.68 [0.59, 0.78] | 0.55 [0.52, 0.59] | 0.81 med |
| 7 Agri. with natural veg. | 237 | 0.67 [0.58, 0.75] | 0.61 [0.57, 0.64] | 0.78 med |
| 8 Agro-forestry | 78 | 0.77 [0.62, 0.91] | 0.50 [0.49, 0.52] | 0.96 high |
| 9 Broad-leaved forest | 299 | 0.58 [0.51, 0.65] | 0.54 [0.50, 0.57] | 0.67 low |
| 10 Coniferous forest | 322 | 0.75 [0.70, 0.80] | 0.57 [0.55, 0.59] | 0.58 low |
| 11 Mixed forest | 367 | 0.76 [0.70, 0.81] | 0.54 [0.52, 0.56] | 0.67 low |
| 13 Moors, heathland | 27 | — | 0.49 [0.48, 0.49] | 0.82 med |
| 14 Transitional woodland | 326 | 0.70 [0.64, 0.76] | 0.50 [0.49, 0.51] | 0.46 low |
| 16 Inland wetlands | 31 | — | 0.50 [0.50, 0.50] | 0.91 med |
| 18 Inland waters | 127 | 0.87 [0.80, 0.93] | 0.74 [0.70, 0.78] | 0.91 high |
| 19 Marine waters | 99 | 0.95 [0.88, 1.00] | 0.61 [0.57, 0.66] | 0.94 high |

- Classes passing E1 (CI lower bound above 0.5):
  - Astra: 12 of 12.
  - Qwen: 10 of 15.
  - Code: 0, since its coverage is 0 on every graded class.
- Reliability, computed without truth, tracks Astra's true BA: Spearman 0.65, p = 0.022, across 12 classes.

## E2: heads on Test

| Class | A+C head BA | AUC | Head − Astra on Test200 [CI] | Qwen head BA | Head − Qwen [CI excl. 0?] | A+C − Qwen heads | True-label head at n=500 | Labels equivalent |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.75 | 0.85 | −0.12 [−0.28, +0.03] | 0.70 | −0.10 | +0.05 | 0.68 | >500 |
| 3 | 0.80 | 0.89 | +0.02 [−0.04, +0.07] | 0.76 | −0.02 | +0.03 | 0.80 | 230 |
| 5 | 0.69 | 0.80 | −0.02 [−0.08, +0.04] | 0.66 | 0.00 | +0.03 | 0.73 | 129 |
| 6 | 0.69 | 0.78 | +0.00 [−0.10, +0.11] | 0.51 | −0.04 | +0.18 | 0.65 | >500 |
| 7 | 0.63 | 0.70 | −0.01 [−0.10, +0.09] | 0.58 | −0.02 | +0.05 | 0.61 | >500 |
| 8 | 0.61 | 0.94 | −0.19 [−0.32, −0.06] | — | — | — | 0.83 | 60 |
| 9 | 0.55 | 0.56 | −0.03 [−0.10, +0.04] | 0.55 | +0.01 | +0.00 | 0.70 | 10 |
| 10 | 0.76 | 0.87 | +0.00 [−0.05, +0.05] | 0.50 | −0.07 | +0.26 | 0.86 | 25 |
| 11 | 0.75 | 0.84 | +0.00 [−0.06, +0.06] | 0.50 | −0.04 | +0.25 | 0.79 | 109 |
| 14 | 0.68 | 0.73 | −0.02 [−0.07, +0.02] | — | — | — | 0.65 | >500 |
| 18 | 0.78 | 0.82 | −0.03 [−0.13, +0.07] | 0.67 | −0.07 | +0.11 | 0.74 | >500 |
| 19 | 0.95 | 0.99 | +0.02 [−0.01, +0.07] | 0.62 | +0.01 | +0.33 | 0.98 | 133 |

Classes 4, 13 and 16 are not Astra-graded. Their A+C heads score BA 0.52, 0.66 and 0.62; the last two beat the 500-label reference, which stays near 0.50–0.58.

- **Gate.** Over the 12 E1-passed classes, the mean of (A+C head − Astra) is −0.031 [−0.058, −0.004]. The requirement was a lower bound above −0.02, so the gate **fails**.
- **Weak to strong.** The Qwen heads do not beat Qwen: mean −0.031 over 11 classes, and no class has a CI above 0.
- **Teacher strength.** The A+C heads beat the Qwen heads by +0.00 to +0.33 BA, with the largest gains on the forest classes and marine water.
- **Label efficiency.** The A+C heads, trained with 0 true labels, beat a head trained on 500 true labels on 7 of 15 graded classes: 1, 6, 7, 13, 14, 16 and 18. Their mean BA is 0.696, against 0.708 for the 500-label reference.

Figures: `figs/e2_teacher_vs_head.png` (teacher BA against head BA, Astra and Qwen marked) and `figs/e2_label_efficiency.png`.

## Why E2 failed (observation, not a design change)
- The gap comes mainly from two rare classes: agro-forestry (−0.19, CI excludes 0) and urban fabric (−0.12).
- Their heads rank well (AUC 0.94 and 0.85) but say yes too rarely at p > 0.5:
  - **Agro-forestry:** head 1.0% yes against Astra 3.5% and truth 6.5%. The head learned from only 17 positive teacher labels.
  - **Urban fabric:** head 10.5% yes against Astra 16.5%.
- With few positives, the logistic heads shrink toward "no". A cutoff set without true labels, for example at the teacher's yes-rate on the bank, would likely close part of the gap. That would be a design change (a CHANGE_REQUEST) and was not run.

## Limits
- Test200 is small for rare classes: urban fabric has 14 positives there, and the CIs are wide.
- The batching check passed exactly at its 80% threshold.
- The code teacher contributed nothing, so "Astra + code" is a single-teacher condition here.
