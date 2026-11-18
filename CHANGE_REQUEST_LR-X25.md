# CHANGE_REQUEST LR-X25: single-patch rejection-sampling fine-tuning (expert iteration)

Status: APPROVED by the user on 2026-11-18 ("let's do that rejection-sampling fine-tuning. just a single sample, see if it works").
This is a diagnostic run. DESIGN_v1.md is unchanged.

- Patch: rlpool index 0 (S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_37_89), arm A. Same patch as LR-X24.
- Each round:
  - Sample 32 DESCRIBE summaries from the current policy (LoRA on; T=1, top_p 1, top_k off; max 160 tokens).
  - Score s with the frozen scorer (LoRA off, layer 21, frozen calibration means).
  - Add the samples to the pool. Target = best candidate of ≤100 words in the pool.
  - Run 30 SFT steps on that one target: AdamW, lr 1e-4, loss on completion tokens only, grad clip 1.0.
- LoRA as in Part B: r16, a32, LM q/k/v/o, dropout 0.
- 4 rounds, then a final evaluation. Logged per round: greedy s, mean and max s of fresh samples, share over 100 words, target s.
- Same-patch references from run_partA.py (rlpool --limit 1): B2 greedy s 0.293 (111 words), A_search s 0.432.
- Code: scripts/rft_single.py. Generation uses HF generate, not vLLM (32 samples, one patch).
