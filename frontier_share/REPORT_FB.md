# Feature bridge (FB-0 to FB-4): report

## Deviations from DESIGN_FB_v1.md

| Id | Deviation | Status |
|---|---|---|
| — | none in the executed design | |

Changes and fixes:
- **FB-X1 (environment, approved).** Codex's read-only sandbox could not run commands on host A because of the AppArmor user-namespace restriction. You approved option 1 and installed a bwrap AppArmor profile. The call settings stayed as frozen.
- **FB-E1 (my audit bug).** The audit read a division in Astra's code (`x/np.sqrt(...)`) as a path and voided A1. The fixed audit finds no flags. All 8 commands only read workspace files, so A1 stands and no retry was spent.

Notes on how I read the design:
- "Seed 0 per target" gives every target the same 5 random bank patches.
- The reader template's entry count follows the number of questions asked.

## Setup
- **Q17.** OLS of mean NDMI on six visible and NIR statistics, fit on the RL pool.
  - R² = 0.59 and the RL-pool yes-rate is 0.456, so the majority answer is no.
  - The Prithvi probe reaches 0.949 CV accuracy (majority 0.544).
  - On the 10 targets, 4 keys are yes.
- **Bank.** 1,000 RL-pool patches across 424 label sets, captioned by Qwen2.5-VL on vLLM in 327 s.
  - Captions average 117 words; 244 stopped at the 160-token limit.
- **Astra calls.** 2 of the 3 allowed: A1 took 211 s and 8 commands, A0 took 196 s and 5 commands. No retry was needed.
- **Neighbor check** (report only): all of each target's 5 Prithvi neighbors share at least one BigEarthNet label with it (share 1.00). The 5 random patches share one 74% of the time.

## 1. Accuracy (skill = accuracy minus majority), 10 targets

| Answerer | visible (100) | shortwave Q11–13 (30) | texture (30) | Q17 (10) | all 17 (170) |
|---|---|---|---|---|---|
| majority | 0.66 | 0.60 | 0.60 | 0.60 | 0.64 |
| Phi-4 C0, caption only | 0.63 (−0.03) | 0.53 (−0.07) | 0.30 (−0.30) | 0.60 (+0.00) | 0.55 (−0.08) |
| Phi-4 C1, + 5 Prithvi neighbors | 0.50 (−0.16) | 0.30 (−0.30) | 0.20 (−0.40) | 0.10 (−0.50) | 0.39 (−0.25) |
| Phi-4 C2, + 5 random patches | 0.54 (−0.12) | 0.40 (−0.20) | 0.30 (−0.30) | 0.30 (−0.30) | 0.46 (−0.18) |
| Astra agent A0, captions only | 0.69 (+0.03) | 0.70 (+0.10) | 0.73 (+0.13) | 0.40 (−0.20) | 0.68 (+0.05) |
| Astra agent A1, + features | 0.73 (+0.07) | 0.70 (+0.10) | 0.60 (+0.00) | 0.40 (−0.20) | 0.68 (+0.05) |
| Astra with images (pilot, Q1–Q16) | 0.77 (+0.11) | 0.63 (+0.03) | 0.63 (+0.03) | — | 0.72 (+0.08) on 16 |
| Prithvi probe (20,000 labels) | 0.86 (+0.20) | 0.77 (+0.17) | 0.90 (+0.30) | 0.90 (+0.30) | 0.85 (+0.22) |

**Phi-4 left many questions unanswered**, and those count as wrong under the frozen convention:
- C1 answered only 100 of 170 questions, C2 answered 131.
- On the 99 questions both answered: C1 scores 0.66, C2 scores 0.59.

## 2. Key contrasts (paired sign test, 170 items)

| Contrast | X right, Y wrong | Y right, X wrong | Acc diff | p |
|---|---|---|---|---|
| C1 − C2 (FM selection) | 11 | 23 | −0.07 | 0.058 |
| C1 − C0 (adding any neighbors) | 7 | 35 | −0.16 | <0.001 |
| A1 − A0 (feature access for Astra) | 7 | 7 | 0.00 | 1.00 |
| A1 − probe (what remains without labels) | 7 | 36 | −0.17 | <0.001 |

## 3. Per-question correct counts (out of 10)
See `fb/readouts_tables.md`. Q17 shows the core result:
- The probe gets 9 of 10.
- A1 and A0 each get 4 of 10.
- C1 gets 1 of 10 and C0 gets 6 of 10, the same as majority.
- The pilot did not ask Astra-with-images Q17, so that row shows 0 there.

## 4. Methods (Astra's own words, abridged; full text in fb/agent_A*.json)
- **A1:** "compared target features with all 1,000 bank features using cosine similarity and Euclidean distance, reviewed nearby captions, and counted relevant land-cover mentions among the 20 closest cosine neighbors. I also inspected feature norms and principal-component positions… Exact indices, CORINE classes, and moisture residuals cannot be calibrated from these unlabeled embeddings alone… every Q17 answer is a low-confidence best guess."
- **A0:** relied on its own captions, plus TF-IDF retrieval of bank captions, which it called weak.
- The commands are listed in `fb/raw_A1_try1/commands.json` and `fb/raw_A0_try2/commands.json`.

## Findings
1. **Without labels, the bridge did not move Prithvi's knowledge into language answers.**
   - Q17 is the clean test: the probe gets 0.90, and every label-free answerer is at or below the 0.60 majority.
   - On the 17 questions overall, the probe beats A1 by 0.17 (p < 0.001).
2. **The Prithvi neighbors are good and Phi-4 cannot use them.**
   - The neighbors share labels with the target 100% of the time, against 74% for random patches. On the 99 questions both conditions answered, neighbors beat random 0.66 to 0.59.
   - But the neighbor block makes Phi-4 stop answering: 70 of 170 answers are "unknown". As designed, C1 therefore scores below C2 and below caption-only.
   - Some of this is the reader's behavior, not missing information. FAB's guided decoding, which allows only yes or no, removes it.
3. **Feature access added nothing for Astra: A1 = A0 = 0.68**, with 7 wins each way.
   - A1 looked at neighbors and principal components but found no way to tie feature directions to the questions without labels, and said so.
   - Both agents beat the pilot's caption-only reader (0.55) because they read their own full structured captions.
4. **The bank captions are weak evidence.** Both agents found errors in them, and they are generic Qwen zero-shot text, a quarter of them cut off at the token limit.

## Limits
- With n = 10 targets, only Q17 and large contrasts mean anything; 7 vs 7 cannot rule out a small A1 effect.
- One reader, one bank, and one random set shared by all targets.
