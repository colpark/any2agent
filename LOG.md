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
