# Naming Prithvi's directions LOG (FN-0 to FN-3)

Times UTC, 2026-11-18/19. Python: ~/langres_state/venv/bin/python. Host A (PCA, diagnostic, Astra) and node A2 (Qwen).

## FN-0
- Read teacher_heads/REPORT_TH.md, teacher_heads/LOG.md and frontier_share/REPORT_FB.md. `DESIGN_FN_v1.md` = section "Frozen design" copied verbatim, sha256 7b2af8fb2a7ae84a6afd551715129cd47fcfa9d2974ce856f3577bdb1356cfcf.
- Codex 0.164.0 "Logged in using ChatGPT". The flags come from teacher_heads th_common.codex_call(agent=False), which are the stage-1 image-QA settings.
- Coverage: the discovery set (1,000 FB-bank RL-pool rows) has finite Prithvi features and 12-band arrays; the held-out set (test, 1,000) has the same.
- Qwen3.5-9B is pinned at c202236235762e1c871ad0ccb60c8ee5ba337b9a (stage 1), and that snapshot is present on A2.
- Code: fn_common.py (reuses th_common: renderings, codex_call, by redirecting its TH/ST folders), fn1_prepare.py, fn2_name.py, fn3_read.py.

## FN-1
- PCA(20, svd_solver full) on the 1,000 × 1,024 discovery features. Explained variance ratios: 0.328, 0.104, 0.078, 0.066, 0.050, 0.032, 0.028, 0.024, 0.021, 0.018, 0.017, 0.014, 0.013, 0.010, 0.009, 0.008, 0.007, 0.007, 0.007, 0.006; sum 0.848.
- End assignment (A end; coin default_rng(d).integers(0,2)): {'0': 'high', '1': 'low', '2': 'high', '3': 'high', '4': 'high', '5': 'high', '6': 'low', '7': 'high', '8': 'high', '9': 'low', '10': 'high', '11': 'low', '12': 'high', '13': 'high', '14': 'low', '15': 'high', '16': 'high', '17': 'high', '18': 'high', '19': 'high'}
- Exemplars: the 6 extreme discovery patches per end, each group shuffled with default_rng(d). Pairs: 40 per direction from the held-out top and bottom 200 (20%), default_rng(d).choice without replacement; position flip default_rng(1000+d).integers(0,2,40).
- Composites: stage-1 pilot renders (th_common.render: frozen stretch, 8 bit, bicubic 512) pasted side by side as 1,024 × 512. 174 discovery + 802 held-out, random file names, in ~/langres_state/fn/renders (433 MB; also copied to A2).
- Note: PC1's A-end exemplars are near-black. Their B04 and B08 means are about 1 (the reflectance floor: dark water or no-data). Checked from the band values; the renders are correct.
- Diagnostic (report only; OLS with intercept on discovery band means and stds): visible = B02, B03, B04, B8A (8 columns); full = plus B11, B12 (12 columns). SWIR gain per PC in fn1.json; range 0.013–0.175.

## FN-2
- `fn2_name.py`: 20 calls n01–n20, 3 concurrent, 12 images each (A1–A6 then B1–B6). Latency 12–42 s. All ok on attempt 1: 0 tool or command items, 0 usage limits, every working dir empty afterwards. Calls used 20/24.
- Word counts (logged, not enforced): names 2–5 words; definitions 35–55 words. names.json written (pc, pc_rank, a_end, name_a, name_b, definition, cues, visible).

## FN-3
- Reader on A2 (`fn3_read.py run`), 1,600 prompts (D1 + D0s × 800 pairs), derangement default_rng(0) over 20; running.
- Reader done on A2: 1,600 prompts in 604 s, 0 unparsed. Prompt hashes per row in reader_answers.jsonl (copied to fm_names/).
- `fn3_read.py grade` -> fn3_measures.json and figs/d1_vs_rank.png.
  - Named: 5/20 (PC 1, 2, 4, 8, 9); named variance share 0.64.
  - Mean D1 0.615 against D0s 0.598, Wilcoxon p = 0.69. Verdict N1 FAILS.
  - Spearman of D1 with PC rank −0.52 (p 0.018), with SWIR gain +0.24 (p 0.31), with visible rating +0.09 (p 0.72). Share of answer "1": D1 0.596, D0s 0.599.
- REPORT_FN.md written. Stopped. No processes left running.
