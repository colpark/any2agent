# AH log (host A = control, Prithvi, embeddings, CPU work; node A2 = bank captions). Times are UTC-5 local, 2026-11-19.

## AH-0
- Read teacher_heads/REPORT_TH.md, teacher_heads/LOG.md, verbal_share/REPORT_VS.md, verbal_share/LOG.md and the stage 1 and VS scripts.
- `DESIGN_AH_v1.md` = section "Frozen design" copied verbatim, sha256 ad5b869b01aeab92ef6ed1635073405c9fb6523465d10453d5cd7b7984fdc08d.
- Codex 0.164.0, "Logged in using ChatGPT". Astra calls for AH: 0 of 4 (planned 2).
- Placement: B2 ssh from host B timed out within 2 minutes (connection timed out), so bank captions run on A2. A2 was idle (115 GB free, no GPU jobs) and host A had no GPU jobs.
- Scripts on A2: vs1.py and vs_common.py are identical to host A (sha256 2297d747…, cd0a0048…); caption_prompt.txt sha256 371e5f9f… and schemas/caption_vs.json copied over; teacher_heads/sets.json identical (5eb1f4b6…).
- Sets (`ah_common.sets`): bank = stage 1 `bank_rl` (500 RL-pool rows); Test = VS `ok_test` (997 of the stage 1 Test); Test200 = stage 1 Test200 inside Test = 199 patches (1 VS-dropped).
- Sources confirmed, no missing answers: Astra bank 500 × 15 and Test200 199 × 15 with confidence (read from the 140 ok raw call outputs in teacher_heads/raw via calls.jsonl; the stage 1 astra_*.json files keep answers only); Qwen bank 500 and Test 997 with confidence (qwen_{bank,test}.jsonl). Qwen bank confidence levels used: high and medium only.
- Class definitions: teacher_heads/code_teacher.json (stage 1 Astra), 15 classes.
- Test200 positives: 1:14, 3:86, 4:7, 5:52, 6:31, 7:45, 8:13, 9:62, 10:68, 11:79, 13:3, 14:69, 16:5, 18:22, 19:22.
- **H1 classes (at least 10 positives in Test200): 1, 3, 5, 6, 7, 8, 9, 10, 11, 14, 18, 19 (12).**
- Code: ah_common.py (shared), ah1.py (inputs, workspaces). Reused unchanged: vs1.captions, vs1.prithvi, vs1.fields, vs_common.descriptors/code_text/caption_fields, th_common.codex_call/audit/check_code. vs1 is pointed at the bank by replacing its `SPLITS` and `ST` (and `VS` for the fields output only).

## AH-1
- 17:51 bank captions on A2: `nice -n 10 python ah1.py captions` (vs1.captions: Qwen3.5-9B c2022362…, bf16, greedy, thinking off, guided JSON, 3,000 tokens, retry 4,500; gpu_memory_utilization 0.6), pid file ~/langres_state/ah/captions.pid (python pid; the first value written was the bash wrapper and was corrected).
- 17:54 bank Prithvi on host A: `nice -n 10 python ah1.py prithvi` (vs1.prithvi: page cache evicted, hooks on 24 blocks, mean over tokens).
- 17:55 bank descriptors (`ah1.py desc`, VS `descriptors` unchanged) for bank 500 and Test 997 -> ~/langres_state/ah/desc_{bank,test}.npy; descriptors.json names, formulas and units.
- Bank Prithvi done: layers_bank (500, 24, 1024) in 8 s. vs1.prithvi's closing check needs a Test file, so ah1 checks the bank instead: the stage 1 normed last token matches rlpool_prithvi_raw.npy for the bank exactly (max abs diff 0.0).
- Code workspace `ws_code` (`ah1.py ws code`): bank_descriptors.npy (500 × 10, raw units), descriptors.json, sample_bands.npy (first 50 bank rows, raw L2A DN float32 as in build_arrays.py), classes.json (15 classes with stage 1 Astra definitions). Label check (no ids or label keys, no column with |r| ≥ 0.99 to a true label): clean. workspaces.json.
- Scaling sentence = stage 1 `th1_code.SCALING` (from build_arrays.py).

- Bank captions done on A2: pass 1 1,169 s, 7 retried at 4,500 tokens, **3 dropped** (failed twice), total 1,504 s. Chat prompt sha256 08453075… and text sha256 371e5f9f… match VS. Raw: ~/langres_state/ah/captions_bank.jsonl (copied from A2). A2 left with no GPU processes.
- `ah1.py fields` (vs1.fields: code field percentiles against the VS Fit descriptors, as for the VS Test fields) -> agent_heads/fields_bank.jsonl; 497 valid.
- `ah1.py embed` (host A, Qwen3-Embedding-4B 5cf2132…, bf16, batch 32, L2-normalized, page cache evicted): (500, 7, 2560), 497 valid, 104 s. The 3 dropped patches stay NaN and leave the bank: **bank n = 497** for every source, fusion and head.
- Free workspace `ws_free` (`ah1.py ws free`): the 12 design files. F standardized with bank statistics; descriptors in raw units; captions = the parsed VS caption JSON; Qwen and Astra answers with confidence for the 15 classes, keyed by class number; test_ids.json holds anonymous row ids t0000–t0996 (no BigEarthNet ids). Each jsonl row carries a row number in file order. Label check clean. The prompt says 500 bank patches; the folder holds 497.

## AH-2
- Astra call 1, code rules (`ah2.py code`, raw/a1_code_try1): ok on the first try, 171 s, 5 commands, audit clean, schema valid, 15 evidence entries (all at most 15 words), static check passed, 119 lines. Prompt sha256 in calls.jsonl. Calls used: 1 of 4.
- Applied in a subprocess (`ah2.py apply`, about 1 s per set, 0 errors). The rules are exclusion-only again: they **never say yes**. They say "no" to land classes on uniform open water and "no" to water classes where vegetation or SWIR reflectance is strong, and abstain otherwise.
  - Coverage (non-abstain share) on the bank: 0.03 for every land class, 0.12 for classes 18 and 19, 0 for class 16. On Test: 0.01 and 0.13. Yes rate 0 everywhere. code_apply.json.
  - So no code single-source head is trainable (needs at least 10 of each answer), and code enters Dawid-Skene only as rare "no" votes. The static check passed, so the design gives no retry.
- Astra call 2, free agent (`ah2.py free`, raw/a2_free_try1): launched 13:21 in a fresh copy of ws_free, sandbox workspace-write, network off (`-c sandbox_workspace_write.network_access=false`), 60-minute limit, venv first on PATH.
  - Audit note: th_common.codex_call flags any file_change item for agent calls. Under workspace-write the design allows writes in the workspace, so ah_common.codex drops that flag only when every changed path lies inside the workspace. Reads are audited as in stage 1.

## AH-3 (dry run while call 2 runs, no free-agent column)
- `ah3.py` 42 s, 12 jobs: 197 heads. Dawid-Skene: fallback to majority vote for classes 8, 13 and 16 (prior below 0.01); 9 classes stop at the 200-iteration cap without meeting the 1e-6 tolerance.
- **Class 14 (transitional woodland) has no fused head.** EM puts its prior at 0.016 (above the 0.01 fallback line) and no bank patch reaches P ≥ 0.5, so the fused label has 0 positives. The design has no rule for this; class 14 is left out of every mean that needs a fused head and reported as missing.
- Fix before the final run: the mean label value now averages the reference curve over the same classes as the fused F mean (it mixed 15 and 14 classes). Trust: the code source scores on almost no class, so per-class Spearman has 2 sources (±1 only); the pooled Spearman over all class × source pairs is added, report only.
- Call 2 result: ok on the first try, 193 s, 9 commands, audit clean; item types agent_message, command_execution, file_change; file changes only `predict.py` and `method.md` inside its workspace (ws_free_run2). scores.npy (997 × 15) valid; method.md 242 words; summary under 100 words. Saved to raw/a2_free_try1: scores.npy, method.md, predict.py, model_diagnostics.json, run.log. Calls used: **2 of 4**, 0 usage limits.

## AH-3 (final)
- `ah3.py` with the free-agent column, 41 s CPU (12 jobs, nice 10). 197 heads.
- AH-E1: the mean label value mixed 15-class and 14-class means (fixed before the final run).
- AH-E2: the report-only extras (Astra-label F head and free agent against the best direct prior on Test200) first crashed on a class-index slip (`startswith("H1")` → `"H1" in k`); no output from that run. Rerun clean.
- Verdicts: H1 fails (+0.012 [−0.013, 0.037], 12 classes, 11 with a fused head); H2 fails (−0.077 [−0.090, −0.062]); substrate passes (F − VC +0.047 [0.029, 0.064], F − D +0.048 [0.033, 0.062]); free agent − fused F +0.082 [0.067, 0.095]. Label value of the fused F head: 30 labels or fewer on average. Trust: pooled Spearman 0.77 (31 pairs).
- Figure: `ah3.py figs` -> figs/fused_vs_prior.png.

## AH-4
- REPORT_AH.md written. No processes left running on host A or A2 (captions, Prithvi, embedding and call PIDs all exited). Stopped per design.
