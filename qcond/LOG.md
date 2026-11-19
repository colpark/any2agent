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
