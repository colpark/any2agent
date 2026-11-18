# Teacher-trained heads LOG (TH-0 to TH-4)

Times UTC, 2026-11-18. Python: ~/langres_state/venv/bin/python (vLLM 0.34.0, transformers 5.22.2, sklearn 1.9.1).

## TH-0
- USER: "stop all others". None of my LangRes, FS, FB or FAB processes were running on host A or A2. Processes from other projects (qa_ladder relay, harbor monitor, v5/v6/v8 mcenv servers) were not started by me and were not touched.
- Read LOG.md, frontier_share/LOG.md and REPORT_FB.md. `DESIGN_TH_v1.md` = section "Frozen design" copied verbatim, sha256 0d98161a94acf8ca5f66ccc2c1178e735bc13567a7d3689ef202b2a663389b77.
- Codex 0.164.0 "Logged in using ChatGPT". `codex sandbox -P :read-only` runs the venv Python with numpy and scipy (FB-X1 fix in place).
- Coverage: the 1,000 FB bank ids map to RL-pool rows with finite cached Prithvi features. Test has 1,000 rows of finite features. Every bank and test patch has at least 1 label.
- Graded classes (at least 20 test positives): 15. Not graded: 2 Industrial (6), 12 Natural grassland (12), 15 Beaches (0), 17 Coastal wetlands (0).
- VLM teacher: `Qwen/Qwen3.5-9B` exists on HF (image-text-to-text, Qwen3_5ForConditionalGeneration, ungated, created 2026-02-27). Pinned revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. vLLM 0.34 lists Qwen3_5ForConditionalGeneration.
  - The download on node A2 started 21:24 with huggingface_hub snapshot_download (pid file ~/langres_state/th/dl.pid).
  - Note: the first pid file landed in A2's ~ (my ssh quoting) and was moved; the ssh session hung on the child's stdin and was closed. The download was unaffected.
- Test300: test split stratified by exact label set (make_splits procedure, default_rng(0), largest remainder) -> teacher_heads/test300.json.
- Code teacher sample: 50 RL-pool rows outside the bank, `default_rng(0).choice(outside, 50)` -> ~/langres_state/th/code_sample.json.
- Scaling sentence filled from build_arrays.py: "raw L2A surface reflectance digital numbers as stored in the BigEarthNet-S2 GeoTIFFs, cast to float32 without rescaling (reflectance × 10,000), with the 20 m and 60 m bands upsampled bilinearly".
- Code check: AST-based. It allows imports of numpy, scipy and math only, rejects the names/attributes open, eval, exec, __import__, os, sys, subprocess, socket (plus compile, globals, locals, getattr, setattr, input, breakpoint), and needs a top-level label(x). Tested: np.cos and a variable named pos pass; import os, open() and np.__import__ fail.
- Code: th_common.py (shared; reuses frontier_share fb3_agent.audit), th1_code.py, th_vlm.py, th2_heads.py, th3_reader.py; 481 lines.

## Prompt update (USER, mid-TH-0)
- The user pasted an updated TH prompt that replaces the first one: Astra becomes the main VLM teacher (142 calls, cap 160), bank 500, Test200, Pilot10 batching check, Qwen one pass, Astra + code labels, a new reliability definition and new gates.
- The first frozen design was kept as DESIGN_TH_v1_superseded_20261118.md (sha256 0d98161a…). The new section "Frozen design" was copied verbatim to DESIGN_TH_v1.md, sha256 9ef002565203b5cefe9e958d8757cc3a74bf80b8b92ec6c2382d4534ddf37b4d.
- The code-teacher spec is identical in both, so calls c001–c002 count under the new cap (160). The 50 sample patches lie outside the FB bank, so they are also outside the 500 bank.
- Sets (th_common.sets, ~/langres_state/th/sets.json and teacher_heads/sets.json):
  - bank 500 = make_splits strat of the FB bank by label set, seed 0;
  - Test300 (as before);
  - Test200 = strat of Test300, seed 0;
  - Pilot10 = the pilot uids.
- Graded: 15 classes on Test (at least 20 positives); for Astra, 12 also have at least 10 in Test200: 1, 3, 5, 6, 7, 8, 9, 10, 11, 14, 18, 19.

## TH-1
- Code teacher, call 1 (`raw/code_teacher_try1`): voided by the audit, 276 s. The verbatim prompt names no Python path; in the sandbox `python` was missing and /usr/bin/python3 had no numpy. Astra then listed /opt, /usr/local/bin and ~ and ran ~/v8env/bin/python. These are real reads outside the workspace, so the void stands.
  - TH-E1 (my setup bug): the design's call setting "the venv Python" was not provided. Fix: run codex with ~/langres_state/venv/bin first on PATH. Sandbox test: `python` and `python3` resolve to the venv and import numpy 2.3.5.
  - TH-E2 (audit false positives): numpy `~` (e.g. `~np.isfinite`) was read as a home path, and `ssh` inside `.venv-ssh` as a network command. The wrapper in th_common drops only these two classes; the real flags of try 1 remain.
- Code teacher, call 2 (`raw/code_teacher_try2`): ok, 212 s, 5 commands, audit clean. Schema valid, 19 definitions, notes ≤150 words, static check passed (81 lines).
  - Astra's notes: "an exclusion-only baseline … it deliberately never emits 'yes'".
  - Applied in a subprocess (`th1_code.py apply`, ~1 s, 0 errors): coverage on the bank is 0.008 for classes 18 and 19 only, and 0 elsewhere; on Test, 0.004 for 18 and 19. It never says yes. So no code-taught head is trainable, and the Astra + code labels equal Astra's except where the code's few "no" answers conflict.
- Qwen3.5-9B smoke test on A2: the first try failed from my script (vLLM spawn re-imported module-level code; fixed with a main() guard). It is not a model failure, so no fallback. Second try: SMOKE_OK, two images described correctly.
- Astra batching check (c003, c004; 39 s each; ok, 0 commands): agreement with the pilot single-patch Q5–Q10 = 48/60 = 0.80. The rule stops only below 80%, so the check passes, at the boundary.
- Qwen teacher on A2 (`th_vlm.py --teacher`, bf16, greedy, enable_thinking False, guided JSON, max_tokens 700): bank 500 in 414 s, test 1000 in 836 s, 0 unparsed. Prompt sha256 6d4c3404495aa028fe61c35bbc6322ea3efcff2ed202987206ac4cbfb4a69ef8.
- Astra bank (100 calls) and Test200 (40 calls): chain `~/langres_state/th/astra_chain.sh`, concurrency 3, renders are 512 px pilot-style with random file names; running.
- Astra chain: bank 100 calls (c005–c104), Test200 40 calls (to c144). All ok on the first attempt, 0 tool items, 0 usage limits. Calls used 144/160.
- Qwen captions for Test300 (A2, `th_vlm.py --captions`, 78 s, 0 hit max_tokens). Prompt sha256 55d49626144fda44d912013761a44a31323173489fa4dcc0eec7cca7415fe6f0.
  - The A2 copy of th_vlm.py still had the pre-rename output name (vlm_captions_test300.jsonl). Copied to ~/langres_state/th/qwen_captions_test300.jsonl; the content is unchanged. These captions are not used, since E3 did not run.
- E1 (`th2_heads.py e1`): gate passes. Passed classes: Astra 12/12; Qwen 10/15 [1, 3, 5, 6, 7, 9, 10, 11, 18, 19]; code 0. Spearman(reliability, Astra BA) = 0.65, p = 0.022. 0 missing answers anywhere.

## TH-2
- `th2_heads.py e2` (91 s, CPU, nice -n 10): 47 heads (Astra, Qwen, A+C; no code head reached 10 yes labels). Label-efficiency reference: n in 10, 30, 100, 300, 500 × seeds 0–2.
- E2 gate FAILS: mean(A+C head − Astra) on Test200 over the 12 E1 classes = −0.031, CI [−0.058, −0.004]; needed a lower bound above −0.02. Weak-to-strong Qwen: mean −0.031, 0/11 classes with CI above 0.
- Diagnostic (report only): yes-rate at p > 0.5 on Test200. Class 8: head 0.010, Astra 0.035, truth 0.065. Class 1: head 0.105, Astra 0.165, truth 0.070.
- Figures: `th4_report.py` -> figs/. STOPPED per the design (E3 not run). No processes left running.
