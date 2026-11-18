# LangRes pilot LOG (prompt: language-aligned z, per-instance search, then RL)

## LR-0 Orientation (2026-11-18 UTC)
- DESIGN_v1.md sha256: `251fe6668103f676c72fb6271f67fd2cd40bc17bac61befb3b7af5c6f4aa23d8` (read-only).
- Earlier repo built from the superseded v1 prompt (orientation + unrun scripts only) moved intact to ~/langres_state/archive_v1_prompt/langres_repo; fresh repo started. Downloads and venv reused.
- host A: GB10 aarch64, 119 GB unified, 284 GB disk free, no foreign GPU procs. Docker yes; no NVIDIA Spark images.
- node A2 (via host A): reachable; 865 GB free; Docker permission denied; no GPU procs.
- host B: reachable; 3.0 TB free; foreign GPU procs VLLM::EngineCore pid 3453169 (68 GB) + ollama pid 1566495 (28 GB) = 96 GB > half -> GPU jobs must wait/poll.
- node B2 (via host B): ssh connect timed out (ConnectTimeout 10 s) at 2026-11-18T04:03Z (also 2026-11-17T22:2xZ). Fallback: check VLM + paraphrase on host A.
- No passwordless sudo on any node; drop_caches root-only -> page-cache drop not permitted (LR-X15).
- Software: uv venv ~/langres_state/venv, torch 2.14.0+cu130 (sm_121 OK), vllm 0.34.0, trl 1.18.1 (RL framework: TRL GRPOTrainer), transformers 5.22.2, peft 0.21.3. Pin: requirements.lock.txt. A2 and B venv installs in progress.
- Layer counts: Qwen2.5-VL-7B-Instruct LM = 28 layers (hidden 3584) -> ℓ = 21. InternVL3-8B LM (Qwen2ForCausalLM) = 28 layers (hidden 3584) -> ℓ' = 21. Readout = HF hidden_states[21] (output of decoder block 21 of 28; index 0 = embeddings) (LR-X16).
- Revisions: model_revisions.json. Llama-3.1-8B-Instruct via mirror (LR-X5).

## Deviations (LR-X)
| id | what | why |
|---|---|---|
| LR-X1 | One uv venv on all nodes instead of NVIDIA Spark containers | no Spark images on any node; A2 user lacks Docker access; same stack on both arms |
| LR-X2 | "Base text model" for perplexity = frozen Qwen2.5-VL-7B-Instruct, text only, summary scored alone | ambiguous; closest to the KL reference |
| LR-X3 | Pool = metadata.parquet (480,038 patches); 69,450 snow/cloud/shadow patches excluded | reBEN recommended set |
| LR-X4 | S2 archive from HF mirror lldewqd/BigEarthNet-S2@47b0b53, accepted only if md5 == Zenodo 2245ed2d1a93f6ce637d839bc856396e | Zenodo capped ~2 MB/s |
| LR-X5 | Llama-3.1-8B-Instruct from unsloth/Llama-3.1-8B-Instruct@4699cc7 (4 safetensors sha256 == meta-llama@0e9e39f) | official repo gated (403) |
| LR-X6 | Stratify by label set = proportional allocation over exact label sets, largest-remainder rounding, seed 0 | exact strata mostly singletons |
| LR-X7 | 60 m bands B01, B09 also bilinear-resampled to 10 m | needed for a 12-band array |
| LR-X15 | Page-cache drop skipped before each GPU job | not permitted |
| LR-X16 | Layer ℓ readout = hidden_states[21] | "layer 21" indexing |

## LR-1 Data (2026-11-18T04:28:18Z)
- metadata.parquet md5 55687065e77b6d0b0f1ff604a6e7b49c (Zenodo record 10891137, v2.0.0).
- Zenodo S2 at ~2 MB/s; HF mirror lldewqd/BigEarthNet-S2@47b0b53 fetched on node A2 (ranged, 12 workers, ~35 MB/s; slow CDN range retried). md5 2245ed2d1a93f6ce637d839bc856396e == Zenodo official -> MD5_OK. 63.25 GB < 150 GB.
- Splits (scripts/make_splits.py, seed 0): calib 200 + rlpool 20,000 (official train, disjoint), val 100 (official validation), test 1,000 (official test). Labels present: calib 17, rlpool 19, val 15, test 17.
- Extracted 21,300 patch dirs (255,600 tifs) on A2 in 1,140 s; rsynced to host A.
- Arrays (scripts/build_arrays.py): 12 bands float32 at 10 m (20 m and 60 m bands bilinear), per-patch sha256 over the 12 raw tif files in band order.
- **Frozen**: ~/langres_data/arrays/splits_frozen.parquet sha256 `47a1d94665106dda3c1ab584dcfa0586539b7a542f625cf109f4644c07cc92fc` (ids + per-patch sha256; read-only). Frozen before any model saw pixels.
- Stretch (2-98 pct, RL pool): {"B02":[37.0,1834.0],"B03":[57.0,1924.0],"B04":[23.0,2250.0],"B08":[1.0,5060.0]}
- Descriptors: descriptors.parquet (mean NDVI/NDWI/NDMI per patch, eps 1e-6 in denominator, + labels).
- Software deviations added: LR-X17 preamble joined to instruction with one space in one text block; LR-X18 rewrite list format "Description k (score x.xxx): w", best first; LR-X19 B1 = "Land cover: " + comma-joined labels + "."; LR-X20 if no search candidate <=100 words, best overall is picked and flagged; LR-X21 check VLM uses the -hf processor's shipped chat template (no system message); LR-X22 InternVL3-8B loaded as official HF-native conversion OpenGVLab/InternVL3-8B-hf@259a3b6 because the original remote code fails under transformers 5.22 (AttributeError all_tied_weights_keys); verified LM 339/339 tensors and projector 6/6 bit-identical, vision differs only by qkv split (340 -> 436 tensors).

## LR-2 Calibration
- Prithvi z cached (~/langres_data/prithvi), standardized with RL pool stats; pos_embed rebuilt (sincos) for num_frames=1.
- LR-E1: run_partA.py vLLM engine failed at start ("An attempt has been made to start a new process before the current process has finished its bootstrapping phase": spawn after HF model initialized CUDA). Fix: `os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")` at top of run_partA.py (in-process engine). No behaviour change.
- LR-E2: layer-sweep diagnostic crashed (sklearn "Input contains NaN"): at hidden_states[0] the last prefilled token's embedding is identical for all patches, so centering gives a zero vector. Fix: skip layers with zero variance (diag value None). Primary layer-21 readout unaffected.
- LR-2 result: calibration AUC (own vs other summary, 200x200, layer 21): arm A 0.800, arm B 0.812 (>= 0.7, pass). Top-1 retrieval A 4.5 %, B 3.5 %. Mean s own A 0.188 / other 0.009; B 0.171 / 0.003. Layer sweep (diagnostic only) in runs/calib_{A,B}/calib_result.json.
- Check VLM (InternVL3-8B-hf, layer 21) calibration AUC: arm A 0.788, arm B 0.745. Means: ~/langres_data/calib/check_means_{A,B}.npz.
- Frozen means: ~/langres_data/calib/means_{A,B}.npz.

## LR-3 Smoke
- A2 arm A attempt 1 failed: "torch.AcceleratorError: CUDA error: out of memory" at model.to(cuda); A2 free RAM 2 GB, 77 GB page cache (our 63 GB archive). Fix (runtime, not design): evict our own files from page cache with posix_fadvise DONTNEED (scripts/evict_cache.py; 68.8 GB evicted; free 2 -> 34 GB), evict model blobs after each load, vLLM gpu_memory_utilization 0.35 -> 0.28.
- Host B: foreign GPU procs hold ~96 GB; arm B smoke waits in gpu_gate.sh polling every 10 min (log: B:~/langres_state/logs/smoke_B.log).
- LR-3 arm A val (100 patches, A2, 1,022 s): s B0 0.037, B1 0.027, B2 0.173, A_search 0.340 (diff vs B2 +0.167 [0.149, 0.187]); s' B0 -0.045, B1 0.046, B2 0.196, A_search 0.259 (+0.063 [0.035, 0.091]); paraphrase of A_search: s 0.340 -> 0.276, s' 0.259 -> 0.221. Full numbers: runs/val_A/readouts_val.json.

## USER-APPROVED CHANGE (2026-11-18): test split reduced 10x
- User instruction: "reduce the test data by 10 fold at least." Test = 100 patches, stratified by label set (largest remainder, seed 0) from the frozen 1,000-patch test split. Split name `test100`; ids ~/langres_data/arrays/test100_ids.json sha256 4e8a5c95131e9ee5f651c15303b5fa8d2ba0a38db1c7f640ba17ffe7b85f5571; 16 labels present. The 1,000-patch membership stays frozen and unchanged.

## USER-APPROVED (2026-11-18): proceed to Part B; evaluate Part B on 10 test patches only
- User: "let's go with part B, on 10 test samples only." Gate 1 waived by user. Training unchanged (450 steps over RL pool, checkpoints every 150).
- `test10` = 10 patches stratified by label set (seed 0) from test100; ids sha256 405c39cbd8cfa0225b94391b6bdb393f074a8c3ec9966d8ac11abc9a9e8cbdfb.
- Arm A RL queued on A2 after the arm A test100 Part A run.
- USER-APPROVED: arm B moved from host B to host A (host B foreign GPU use ~96 GB > half since 2026-11-18T04:56Z). Host B waiter stopped (own PID file). Arm B val smoke skipped (code path validated by arm A smoke; LR-X23); arm B runs test100 Part A, then RL, on host A.
- 2026-11-18 ~05:44Z USER: "interrupt part A and start RL immediately." Killed own run_partA.py (A2 pid 1706550, host A pid 2430590) during test100 search round 0. Completed before kill: test100 pass 1, B0, B1, B2 (arm A); pass 1, B0, B1 (arm B). A_search on test100 not done.
- LR-E3: first RL launch failed on both arms in 10 s: "TypeError: `train_dataset` must be a `Dataset` or `IterableDataset`, got `PatchDS`" (TRL 1.18). Fix: datasets.Dataset of indices with set_transform building the image prompt lazily. 2-step test run on host A (runs/rl_test) before relaunch.
- RL relaunched after LR-E3: arm A on A2, arm B on host A, 450 steps each. Test step: 292 s/step on host A (arm B) -> ~36 h per arm projected.

### 2026-11-18 12:05 UTC, RL monitor
- Arm A (A2): step 131/450, 165 s/step, reward -0.13 -> +0.17, KL 0.045, mean length 104 tokens, clipped 0.3%. Rollout s (T=1) 0.165 -> 0.216; >100 words 60% -> 9%.
- Arm B (host A): step 75/450, 285 s/step, reward -0.31 -> +0.03, KL 0.046, length 106 tokens. Rollout s 0.079 -> 0.094; >100 words 79% -> 12%.
- No reward-hacking sign in rollouts: commas per word flat (0.07), text stays prose. s' is checked at the step-150 checkpoints.

### 2026-11-18 ~13:00 UTC, RL diagnosis (no change made)
- Step timing from log-file growth. Phase 1 = generation + reference pass + reward; phase 2 = policy forward/backward.
  - Arm B: phase 1 123 s, phase 2 162 s.
  - Arm A: phase 1 73 s, phase 2 90 s.
- Correction to the 06:05 note: generation is not "nearly all" of the step. Training is the larger part. vLLM generation could save at most ~40% of arm B's step.
- Reward decomposition, within-group variance: the length penalty supplies ~70% of the reward variance from step 30 on, although only ~10% of rollouts exceed 100 words. s supplies ~30%.
- s of rollouts with ≤100 words, per 15-step bin (SE ≈ 0.007):
  - Arm A: 0.170 → 0.215 (gain mostly after step 100).
  - Arm B: 0.073 → 0.090.
  - corr(s, words) ≈ 0.
- grad_norm ≈ 0.07 (clip 1.0); KL flat at ≈ 0.045 since step 30.

### 2026-11-18 LR-X24 / LR-X25 (user-approved diagnostics), full RL runs stopped
- The user chose "Stop both full runs". rl_A stopped at step ~140 and rl_B at step ~80, by own PID files. Logs kept in runs/rl_{A,B}_stopped1. No checkpoint was reached.
- LR-X24 single-patch GRPO (rlpool idx 0, lr 1e-4, graded penalty, 32 samples/step, 150 steps) is running on host A at ~50 s/step.
- Same-patch baseline from run_partA (vLLM): B0 -0.337, B1 -0.223, B2 0.293 (111 words), A_search 0.432.
- LR-X25 single-patch RFT on A2, 194 s total:
  - Greedy s by round: 0.363, 0.403, 0.439, 0.439, 0.439. Targets 0.407, then 0.439.
  - Sample mean s: 0.328 → 0.434. Samples over 100 words: 84% → 0%.
  - From round 2 on, all 32 samples are near-identical (max 0.434) and SFT loss is ~0, i.e. memorized. It stalls at the best ≤100-word sample seen (the 0.506 sample had >100 words).
  - Note: HF greedy (0.363) differs from vLLM greedy (0.293) on the same patch and prompt. The decoders differ numerically; both summaries have 111 words.

### 2026-11-18 LR-X24 vs LR-X25 head-to-head, rlpool idx 0 (scripts/eval_single.py; runs/eval_single_idx0.json)
- Setup: one generator (HF generate), greedy + 32 samples at T=1, seed 0. Scorers: Qwen s (LoRA off) and InternVL s'.
- Greedy s / s':
  - base 0.361 / 0.414
  - RFT 0.435 / 0.384
  - RL50 0.368 / 0.443
  - RL100 0.514 / 0.462
  - RL150 0.501 / 0.443 (101 words)
  - A_search 0.431 / 0.412
  - vLLM B2 0.290 / 0.326
- Samples, mean s / mean s' / distinct of 32 / within-set Jaccard:
  - base 0.328 / 0.329 / 32 / 0.24
  - RFT 0.435 / 0.383 / 1 / 1.00
  - RL150 0.482 / 0.456 / 32 / 0.35
- RFT greedy gains +0.07 on s but loses 0.03 on s' versus base greedy: the target was picked by s on one patch, so it does not transfer.
- RL gains on both scorers. corr(s, s') within the RL samples falls from 0.77 (base) to 0.17–0.42; watch this.
- Wall time: RFT 194 s; RL 150 steps ~1 h 55 min.

### 2026-11-18 LR-X26 prior-ensemble pilot started (CHANGE_REQUEST_LR-X26.md, user-approved)
- Downloads, pinned in model_revisions.json:
  - LLaVA-OV-7B 0d50680
  - Idefics3-8B fddb4ff
  - Phi-4 2db69c1
  - Pixtral-12B c2756cb (on A2; HF redirects it to mistral-experimental/pixtral-12b)
  - All ungated; licenses Apache-2.0 / MIT. No new packages were needed.
- Writers on host A: LOV and IDF. Writer on A2: PIX. A pull script copies the PIX captions to host A. Fusion and scoring run on host A.
- LR-E4: the first LOV launch failed with a CUDA OOM at context creation. Page cache from the concurrent downloads (GB10 unified memory) caused it. Fix: ensemble.py evicts our own HF cache and data before loading, and the chain retries.
- Idefics3 runs with do_image_splitting=False: the 120 px source is upsampled to 448, so tiling adds nothing.
- E2 uses 3 samples per writer, so K matches E3 with all 3 writers.

### 2026-11-18 LR-X26 results, val n=100, arm A (runs/ens_val/ensemble_readout_val.json; raw captions in runs/ens_raw)
- Raw captions, s / s':
  - LOV 0.145 / 0.220
  - PIX 0.134 / 0.159
  - IDF 0.055 / 0.080
  - concat 0.131 / 0.078
  - best-of-3 0.207 / 0.216
- Phi-4 K=1 rewrite (E1), s / s':
  - QWEN 0.147 / 0.195; the raw Qwen B2 was 0.173 / 0.196. The rewrite removes Qwen's s style advantage and keeps s'.
  - LOV 0.122 / 0.192
  - PIX 0.120 / 0.158
  - IDF 0.050 / 0.080
- E2 (one writer, 3 samples fused), s / s': LOV 0.131 / 0.160; PIX 0.090 / 0.094; IDF 0.059 / 0.057.
- E3 union fusion, s / s': LOV+PIX 0.136 / 0.165; IDF+LOV 0.100 / 0.115; IDF+PIX 0.077 / 0.071; all three 0.088 / 0.105.
  - Phi-4 exceeded 100 words in 49–65% of union outputs.
- E4 consensus fusion of all three: 0.104 / 0.147. E5 concat: 0.109 / 0.110. E6 best-of E1 by s: 0.190 / 0.210 (vs B2: +0.017 / +0.014, n.s.).
- Coverage curve (mean over subsets), s': 0.143 (1 writer), 0.117 (2), 0.105 (3). It falls with more priors.
- Single-patch adapters on val (vLLM greedy), s / s':
  - RL1p (sp_A ckpt-150): 0.154 / 0.117 (s' -0.079 vs B2).
  - RFT1p: 0.037 / 0.007. It emits the memorized patch-0 summary for every patch.
