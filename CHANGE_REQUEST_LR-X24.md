# CHANGE_REQUEST LR-X24: single-patch GRPO sanity test

Status: APPROVED by the user on 2026-11-18 ("use a single patch and about 150 steps with all the design changes").
This is a diagnostic run. The frozen design (DESIGN_v1.md) is unchanged, and the full-design runs keep their defaults.

| Item | Frozen design | LR-X24 test |
|---|---|---|
| Patches | RL pool, 32 per step, each used once | one patch (rlpool index 0), repeated every step |
| Samples per step | 32 × 4 = 128 | 4 groups × 8 = 32 |
| LoRA LR | 1e-5 | 1e-4 |
| Length penalty | 0.5 × [words > 100] | min(0.5, 0.05 × (words − 100)) for words > 100 |
| Steps | 450 | 150, saved every 50 |
| Everything else | unchanged | unchanged (beta 0.04, clip 0.2, r16, T=1.0, scorer = adapter off) |

Arm A only (one image, the faster arm).
Reference points for the same patch: B2 zero-shot greedy and A_search, from run_partA.py on rlpool --limit 1.
Read-out: per-step s of rollouts; greedy s and InternVL s′ at steps 50, 100 and 150.
