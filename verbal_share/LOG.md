# VS log (host A = control and Prithvi/embeddings; node A2 = captions). Times are UTC-5 local.

## VS-0 (2026-11-18)
- Read REPORT_FN, REPORT_TH, REPORT_FB, REPORT_FS_PILOT. Reused: th_common.strat (make_splits logic), scripts/prithvi.py (exec'd unchanged up to run()), scripts/common.arm_images (arm B, 448 px), fs2_astra CAPTION_PROMPT, frontier_share/schemas/caption.json, FB q17.npz OLS.
- DESIGN_VS_v1.md sha256 e0e36616ddec4fec4dfd572f8f4592c648437695ec8be777a6b4b87029677b26.
- `python scripts/vs0_prepare.py` -> vs0.json, sets.json: Fit 2,000 RL-pool patches over 641 label sets (seed 0); Test 1,000. Graded classes (>= 20 positives in both): 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19 (15). Q17 key matches FB on the RL pool and Test100 (1.0); yes rate 0.463 Fit, 0.478 Test.
- schemas/caption_vs.json sha256 c5c7acb5…; caption_prompt.txt (pilot prompt with the summary sentence replaced; exact replacement asserted) sha256 371e5f9f14b5879b35dd8e51db7f4d455523a2a4481e10d51734a19ddfd2b5a4. Images: arm B true color + false color, no extra preamble (the prompt itself names image 1 and image 2).
- Revisions: Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a (stage 1, confirmed on the Hub); Qwen/Qwen3-Embedding-4B 5cf2132abc99cad020ac570b19d031efec650f2b (downloaded to host A, 62 s); fallback Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; Prithvi-EO-2.0-300M 9eb1b110…
- Placement: B2 ssh from host B timed out again, so captions run on A2 only.

## VS-1
- `vs1.py prithvi` (host A, nice 10, pid file prithvi.pid): layers_fit (2000, 24, 1024) 28 s, layers_test (1000, 24, 1024) 13 s. Hooks on blocks 1-24, CLS dropped, mean over tokens. Block 24 hook = pre-final-norm output. Stage-1 normed last token recomputed alongside: matches test_prithvi_raw.npy exactly (max abs diff 0.0).
- Caption smoke (VS_SMOKE=4, A2): default guided decoding at 1,000/1,500 gave 2/8 valid (pretty JSON, length stops).
- VS-E1 (attempted): set `structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True}` on the engine (the per-request flag is ignored; engine config showed disable_any_whitespace=False). Result: valid but degenerate captions (empty arrays). To be reverted under CR VS-X1 option 1.
- Probe for the CR (probe_lengths.py, 16 patches, A2; vLLM spawn needed a main() guard): see CHANGE_REQUEST_VS-X1.md. Waiting for approval; no design captions run.
- CR VS-X1 approved (option 1) 2026-11-19. vs1.py: VS-E1 reverted (default guided decoding), max tokens 3,000, retry 4,500. First launch failed on a syntax slip in the retry line (fixed before any output).
- Caption try 1 (A2, pid 1785707) stalled: node at 118/119 GB with other long-running jobs; prompt rendering 10 s/patch. Stopped by own PID file; its orphaned vLLM engine (1785975, 72 GB) and resource tracker stopped too. No captions kept (log captions_try1_killed.log).
- At the user's request ("stop whatever has been running if you did not start and is not relevant to this study"), stopped on A2: 23 `v5env ... -m mcenv.server*` processes (27-39 days old) and a 9-day watcher loop for `extract.py --prior p2 --task t2` (decompose project); on host A: `v8env server8.py serve --port 8881`, `qa_ladder_runs/.../brain_probe_relay.py`, `harbor/monitor/node_monitor.py`, and three 40-day-old Claude shell wait loops. Left desktop apps (Firefox, LibreOffice, Xorg). A2 after: 115 GB free, no GPU jobs.
- Caption try 2 launched on A2 (pid 1789279), gpu_memory_utilization 0.6.
- VS-2 dry run on synthetic embeddings (dry_*.npy, VS_EMB=dry) to time the code; its outputs are overwritten by the real run.
- Captions done (A2, try 2): Fit pass 1 4,392 s, 17 retried at 4,500, 3 dropped; Test pass 1 2,302 s, 8 retried, 3 dropped. Chat prompt sha256 0845307…; text sha256 371e5f9…. Raw outputs: ~/langres_state/vs/captions_{fit,test}.jsonl (copied to host A).
- `vs1.py fields`: valid Fit 1,997, Test 997. Code field follows the frozen template verbatim ("{p}th", e.g. "83th").
- VS-2 speed fix (no change in results): kernel CV used scipy cdist (single-threaded) per fold and gamma; replaced by one BLAS squared-distance matrix per fold. Dry run: identical numbers, 5x faster (full VS-2 2,283 s on synthetic inputs).
- `vs1.py embed` (host A, Qwen3-Embedding-4B 5cf2132…, bf16, no prefix, batch 32, L2-normalized) running.
- Embeddings: fit (2000, 7, 2560) 403 s, test 612 s; L2 norms 0.996-1.004 (bf16). 6 dropped patches stay NaN and are masked in VS-2/VS-3.
- `vs2_maps.py` (host A, nice 10): 2,367 s. Selected map: kernel (VC CV R2 0.715). Outputs vs2.json, fhat/resid npy.
- `vs3_probes.py` (host A, 12 jobs): 514 s. Outputs vs3.json, figs. Figure layout fixed twice (legend placement, minor ticks) with `vs3_probes.py figs`; numbers unchanged.
- REPORT_VS.md written. Stopped per design.
