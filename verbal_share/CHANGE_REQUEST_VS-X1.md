# CHANGE_REQUEST VS-X1: caption token budget

**Status:** approved 2026-11-19 (option 1). Max new tokens 3,000, one retry at 4,500, default guided decoding.

## Problem
The design caps Qwen3.5-9B captions at 1,000 new tokens and retries once at 1,500. With the frozen prompt and schema, almost no caption fits in that budget.

Evidence, from 16 smoke patches (the first 8 Fit and the first 8 Test), with outputs in `~/langres_state/vs/probe_{a,b}.jsonl`:

| Decoding | Max tokens | Valid | Tokens used | Content |
|---|---|---|---|---|
| Default guided JSON (vLLM auto backend, pretty-printed) | 1,000, then 1,500 | 2 of 8 (first smoke test) | 1,186 to over 1,500 | full |
| Same, uncapped | 4,000 | **16 of 16**, all stopped naturally | 1,184 to 2,677, median about 1,600 | 9 grid cells, 1 to 24 features, 3 to 7 claims, 58 to 92 summary words |
| Same JSON re-serialized compact (tokens counted afterwards) | n/a | n/a | 715 to 1,460; 9 of 16 over 1,000 | full |
| Compact decoding (`disable_any_whitespace`), frozen schema | 1,000 | 8 of 8 | 168 to 333 | **degenerate**: empty cover, grid, features and claims arrays |
| Compact decoding, schema with `minItems` added (grid exactly 9) | 1,500 | 12 of 16 | 636 to 1,239; 4 truncated | full but thinner |

- Pretty-printing costs about 40% extra tokens.
- Even the compact content alone exceeds 1,000 tokens for most patches.
- Compact decoding with the frozen schema makes the model close every array at once.
- Holding the budget would drop most patches or keep nearly empty captions.

## Options
1. **(Recommended) Raise the budget only.** Keep the prompt, the schema and the default guided decoding unchanged. Set max new tokens to **3,000**, with one retry at **4,500** (the prompt plus 4,500 fits in `max_model_len` 8,192). On the probe, all 16 patches would pass on the first try.
   - Cost: about 1,700 tokens per patch, roughly 5M output tokens for 3,000 patches. I estimate 1.5 to 2.5 hours on A2 alone (B2 is unreachable).
2. **Compact decoding plus `minItems` in the schema** (grid exactly 9; cover, features, implications, interpretation and uncertainties at least 1), with a 1,500 budget and a retry at 2,500.
   - Fewer tokens, but it changes the schema, and on the probe 4 of 16 still truncated at 1,500.
3. **Keep v1 as written.** Most patches would be dropped, so VS-2 and VS-3 would run on a small biased subset. Not recommended.

## Also logged (no approval needed)
VS-E1 tried to enforce compact JSON (`disable_any_whitespace`) at the engine level, believing that was the v1 intent. That turned out to cause the degenerate outputs. Under option 1, I revert vs1.py to the default decoding exactly as the design states it ("JSON-schema guided decoding").
