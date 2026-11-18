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
