# QC log (host A: Astra calls, embeddings, CPU work). Times are local, 2026-11-19.

## QC-0
- Read agent_heads/REPORT_AH.md and LOG.md, verbal_share/REPORT_VS.md, teacher_heads/REPORT_TH.md, and the AH, VS and TH scripts.
- `DESIGN_QC_v1.md` = section "Frozen design" copied verbatim, sha256 159ef6ca29a988f8ce1bae28b838a703c6808aab53d1848cf919b6a63c2e3aef.
- Codex 0.164.0, "Logged in using ChatGPT", model gpt-6-astra. Host A idle (no GPU processes). Other nodes not used.
- `qc0.py` (nice 10): Practice = `default_rng(0).choice(497, 300, replace=False)`, sorted, over the AH bank rows in file order -> practice.json. F Practice (300, 4096) and Test (997, 4096), standardized with Practice statistics; VS Test field embeddings (997, 7, 2560), no NaN; stage 1 Astra answers (19 classes, with confidence) complete for Practice and Test200 (199); Qwen Test answers complete; AH descriptors bank (500) and Test (997).
- Implementation choice (not a design change): G1–G8 derived on scores use OR = max, AND = min, "more than three" = 4th largest member score; on 0/1 answers this is exactly the stage 1 key rule.

## QC-1
- `qc1.py keys` (CPU): 31 physical questions and exact keys on Practice and Test -> physical.json, ~/langres_state/qc/phys_{practice,test}.npy. Percentiles from the 497 bank descriptors. P3 OLS refitted on the full RL pool: R² NDMI 0.592, B11 0.891, B12 0.846; the Q17 key reproduces FB's q17.npz on the RL pool exactly. P2 halves: columns 0–59 vs 60–119 (left/right), rows 0–59 vs 60–119 (top/bottom).
  - P1_5_bottom (water share in the bottom quarter) is never yes (the 25th percentile of water share is 0): yes rate 0 on Practice and Test. It is a practice question and will be skipped by the answerer rule (fewer than 10 of either answer).
- Astra call q001 (question writing, `qc1.py ask`, PID 3507302): **voided_tool**, 237 s. Astra ran `cat ~/.codex/skills/qa-ladder/SKILL.md`.
- Astra call q002 (retry, unchanged settings): **voided_tool**, 267 s. Same skill file plus references/setup.md. Output (150 questions) kept in raw/q002_ask and not used.
- Calls used: 2 of 140, no usage limit. **Stopped: CHANGE_REQUEST_QC-X1.md awaits approval.**
- QC-X1 approved. `qc_common.call` takes extra codex flags (inserted after `exec`); the question call passes `--enable skip_host_skill_discovery`. The ledger `cmd` field is built before the shim and does not show the flag; the run's events confirm it was applied (Codex emitted "Under-development features enabled: skip_host_skill_discovery").
- Astra call q003 (question writing, with the flag): **voided_tool**, 286 s. Astra again ran `cat ~/.codex/skills/qa-ladder/SKILL.md`, then references/setup.md and inputs.md. The flag does not hide the skill.
- Calls used: 3 of 140, no usage limit. **Stopped per QC-X1** (no more retries without approval). QC-2 to QC-4 not started; QC-5 (add-on) waits for QC-4.
- User reported moving the skill out. QC-X1 flag removed (back to the frozen settings). Astra call q004 (question writing): **voided_tool**, 349 s; Astra read `~/.codex/skills/qa-ladder/SKILL.md` and references/setup.md again. My error: the `ls ~/.codex/skills` run just before the call already showed `qa-ladder` still on host A, and I launched without reading it. On host A the folder still holds `qa-ladder` (dated Nov 17) plus five hidden `.qa-ladder.bak-*` copies.
- Calls used: 4 of 140. Stopped.
- 17:56 With the user's go-ahead I moved ~/.codex/skills/qa-ladder and the five .qa-ladder.bak-* copies to ~/.codex/skills_off/ (restore after QC). ~/.codex/skills now holds only .system.
- Astra call q005 (question writing, frozen settings, no extra flag): **ok**, 216 s, no tools, 150 questions -> generated.json (raw/q005_ask). Prompt sha256 in calls.jsonl.
- `qc1.py embed` (GPU host A, nice 10, page cache evicted 424.7 GB): Qwen3-Embedding-4B 5cf2132…, bf16, no prefix, normalized; 204 texts (23 held-out semantic, 31 physical, 150 generated) in 3 s -> ~/langres_state/qc/emb_all.npy.
- `qc1.py build`: leakage filter drops none (max cosine to a held-out question: min 0.461, median 0.570, max 0.712); first 120 kept (A1–A120) -> questions.json: 141 practice (120 semantic + 21 physical), 33 held out (23 semantic + 10 physical). leakage_filter.json.

## QC-2
- Order check (`qc2.py check`, q006–q009, group 1, halves A and B, given and reversed order): all ok, 103–192 s, no tools. Agreement 0.913 over 600 answers -> pass (order_check.json). Calls used 9 of 140. Renders: stage 1 render() (RL-pool 2–98% stretch, bicubic to 512) into ~/langres_state/qc/renders.
- 18:05 main answer calls launched: `setsid nohup nice -n 10 python scripts/qc2.py main` (118 calls, groups 2–60 × halves A and B, concurrency 3, one retry on invalid or voided), python PID 583131 in ~/langres_state/qc/main.pid (the first value written was the shell wrapper and was corrected).
- While the calls run: qc3.py, qc4.py and qc5.py dry-run on placeholder labels (order-check answers plus random fills; QC_DRY=_dry, outputs under ~/langres_state/qc and discarded). All three run end to end (qc3 58 s); the QC-5 logit identity check passes on the dry formula.
- 19:30 main calls done: 118 of 118 ok on the first try (q010–q127), no tools, no usage limit. Calls used: **127 of 140** (123 planned + 4 voided question calls).
- `qc2.py soft`: 300 × 120 soft labels, none missing. Yes rates (P ≥ 0.5) per question: min 0.02, median 0.41, max 0.79. Low-confidence answers are rare (216 of 36,000).

## QC-3
- `qc3.py` (CPU host A, nice 10, 12 jobs, 51 s): 139 answerers (skipped by the 10-answer rule: P1_5_bottom, A113). C = 0.01 (mean CV log loss 0.328, 0.405, 0.574 for C = 0.01, 0.1, 1). Ridge alpha = 0.316 (leave-one-question-out mean AUC 0.783; 0.856 on 119 semantic, 0.352 on 20 physical). T = 1.86. Held-out scores, language route, Astra and Qwen direct, reference heads and true-label heads (n = 30, 100, 300; seeds 0, 1, 2) -> ~/langres_state/qc/qc3.npz, qc3.json.

## QC-4
- `qc4.py` (CPU, 1,000 bootstrap resamples) -> qc4.json, figs/generalization.png, figs/cascade.png.
- **R1 passes:** +0.031 [0.015, 0.048] (QC 0.672, language 0.641). **R2 fails:** 0.816 of the reference heads (0.824). **Cost fails:** Astra direct mean BA 0.731; QC alone 0.573; τ = 0.05 already sends 77% of Test200 patches to Astra.
- Seven held-out questions have QC AUC below 0.5 (C8, C19, P1_6_bottom, P1_9_top, P2_moister_lr, P2_moister_tb, P3_ndmi). Their nearest practice questions are the opposite-polarity twins (top/bottom quarter cosine ~0.9).
- REPORT_QC.md written.

## QC-5 (add-on)
- `DESIGN_QC5_addendum.md` = section "Frozen addendum" copied verbatim, sha256 4b097850addb84a2c765dbc472fca3e66c78809037a5fb5e89430ae3b18fb26c. Reads v1 artifacts only; no v1 file edited.
- `qc5.py` (CPU, nice 10): logit identity check max abs diff 4.9e-15 (pass). Caption ridge alpha = 1000 (top of grid).
- A: R@1 / R@10 / median rank: QC 0.006 / 0.041 / 301; caption ridge 0.008 / 0.054 / 234; shuffled control 0.002 / 0.010 / 504.
- B: 997 patches, 0 skipped; QC 0.761, language 0.684, diff +0.077 [0.062, 0.091]. Effective rank of M_w 138.
- REPORT_QC5.md written. No processes left running. The Codex skill stays parked in ~/.codex/skills_off/ until the user restores it.
