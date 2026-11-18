# Frontier share pilot LOG (FS-0 to FS-4)

Host A only. Python: `~/langres_state/venv/bin/python` (LangRes venv; sklearn 1.9.1, jsonschema 4.26.0). Times UTC.

## FS-0 (2026-11-18)
- Read LOG.md, DESIGN_v1.md, scripts/common.py, build_arrays.py, prithvi.py.
- `frontier_share/DESIGN_FS_v1.md` = section "Frozen design" copied verbatim. sha256 d7c7004675ea4063bf1767321ecf881dd595b5118db6631faf66b12a1b34765a.
- Schemas written strictly from the design (every object: additionalProperties false, all properties required):
  - schemas/caption.json sha256 5b0cd65b73c7041a7a0454196f09d453c6237c0c17660680c81ec01f99110683
  - schemas/qa.json sha256 eff36cbc2e3fb7bcd5cab5c003ff2a3908ed8dcf9ef35eb85fa3debdddc7ae67 (answers: id string, answer yes|no, confidence low|medium|high; count and order of ids checked in code)
- Codex: `codex --version` -> codex-cli 0.164.0 (already installed at ~/.local/bin/codex). `codex login status` -> "Logged in using ChatGPT". The login already existed, so no device login was needed. No credential was read or handled.
- Confirmed from `codex exec --help`, `~/.codex/config.toml` and `~/.codex/models_cache.json`:
  - Model string: `gpt-6-astra` (display name GPT-6-Astra; input modalities text, image; supported efforts low..ultra, includes high). Passed as `-m gpt-6-astra`.
  - Image flag: `-i, --image <FILE>...`.
  - Output schema: `--output-schema <FILE>`.
  - Last message to file: `-o, --output-last-message <FILE>`.
  - JSON event stream: `--json` (JSONL on stdout).
- Call settings: `codex exec --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check -m gpt-6-astra -s read-only -c approval_policy="never" -c model_reasoning_effort="high" -c web_search="disabled" --output-schema S --json -o F -C <fresh empty dir> -i tc.png fc.png`, prompt on stdin.
  - `--ignore-user-config` keeps the user's plugins and project trust settings (mcbench plugin, xhigh default effort) out of the calls; auth still comes from CODEX_HOME.
  - Config values checked locally with `codex -c ... mcp list` (makes no model call): web_search accepts disabled|cached|indexed|live; approval_policy accepts never. No search flag is passed.
  - Working dir: fresh empty `~/langres_state/fs/tmp/<call>_<rand>/`. The images are attached from frontier_share/renders.

## FS-1
- `frontier_share/scripts/fs1_prepare.py` (18 s). Statistics come from the 12-band arrays: RL pool `rlpool_ms12.npy` (20,000) and test100 rows of `test_ms12.npy`. Per pixel: ε = 1e-6 added to the denominator.
- Frozen thresholds (RL pool medians, rounded once and used in both text and key): NDMI 0.192, B08 std 640.0, B08 edge 178.0. Edge = mean |Δ| over all horizontal and vertical B08 neighbor pairs.
- Top-6 RL pool labels (Q5..Q10): Arable land; Mixed forest; Coniferous forest; Transitional woodland, shrub; Broad-leaved forest; Land principally occupied by agriculture, with significant areas of natural vegetation.
- Majority baseline (RL pool yes-rate): Q1 yes (0.585), Q2 no (0.177), Q3 no (0.288), Q4 no (0.495), Q5..Q10 no (0.25..0.37), Q11 yes (0.5002), Q12 no (0.210), Q13 yes (0.503), Q14 yes (0.5001), Q15 no (0.328), Q16 no (0.49998).
- Selection: tertile cuts NDVI 0.444/0.700, NDMI 0.085/0.262. Cell population (ndvi,ndmi): 00:21 01:5 02:7 10:12 11:19 12:2 20:0 21:9 22:25.
  - Cell (2,0) (high NDVI, low NDMI) is empty, so it was filled from the most populated cell (2,2), as the design states. Final: 8 cells x 1 + cell (2,2) x 3 (its 3 lowest hashes).
- Anonymous ids: `uuid.uuid4()` after `random.seed(0)`, as written. Note: uuid4 draws from os.urandom, so the seed does not make the ids reproducible. The map is saved in frontier_share/private/id_map.json (gitignored). It has no effect on any result.
- Renderings: frontier_share/renders/{uid}_{tc,fc}.png. The frozen RL pool stretch is applied first, then 8-bit, then PIL bicubic 120 -> 512 (gitignored).
- Keys: ~/langres_state/fs/keys_{rlpool,test100}.npz. Yes-rate over the 10 selected: Q1 .7 Q2 .1 Q3 .3 Q4 .3 Q5 .3 Q6 .8 Q7 .5 Q8 .4 Q9 .2 Q10 .2 Q11 .6 Q12 .2 Q13 .4 Q14 .7 Q15 .2 Q16 .7.
- Page cache of ~/langres_data/arrays evicted afterwards (own files).
- Probes: `scripts/fs1_probes.py` -> probes.json. Cached RL-pool-standardized Prithvi features (rlpool 20,000 x 1024; test100 rows checked aligned with test100_ids). LogisticRegression(C=1.0, max_iter=3000), 5-fold stratified CV (shuffle, seed 0), then a full fit, then a prediction on the 10. No test features were missing.
  - CV acc (majority) per question: Q1 .985 (.585), Q2 .990 (.823), Q3 .964 (.712), Q4 .711 (.505), Q5 .881 (.629), Q6 .842 (.668), Q7 .901 (.679), Q8 .787 (.731), Q9 .855 (.731), Q10 .815 (.747), Q11 .974 (.500), Q12 .978 (.790), Q13 .701 (.503), Q14 .952 (.500), Q15 .768 (.672), Q16 .960 (.500). All converged under 3000 iterations.

## FS-2
- `scripts/fs2_astra.py` (ledger: calls.jsonl; raw: raw/cNN_*/{events.jsonl,last_message.json,stderr.txt}; validated outputs: raw/{caption,qa}_{role}_{uid}.json).
  - Prompts are verbatim from the design. QA question block: one line per question, "Qn. text", in order.
  - Prompt sha256: caption 97e6f8f616577022b52cd6660734012695fdbc8fb2ed4014cc8281653aa89699 (20 calls), QA d9dd4d8a3ccf1222f94f46b2c2d446e5ac775c952d7f609c390317664aaea5e1 (10 calls).
- Schema test: c01, c02 = main captions of the first 2 selected patches. Both valid (schema + word budgets), 128 s and 122 s. The event stream shows only thread.started, turn.started, one agent_message and turn.completed.
- Chain `~/langres_state/fs/fs2_chain.sh` ran in this order: 8 main captions (c03–c10), 10 retest captions (c11–c20), 10 image QA (c21–c30). Concurrency 3.
- Result: 30/35 calls used. 30 ok, 0 invalid, 0 retries, 0 tool or command items, 0 usage-limit messages. Every working dir was still empty afterwards.
  - Caption latency 107–140 s (mean ~124 s); QA 13–23 s.
  - Tokens per caption: ~13.4k–14.3k input, 4.1k–5.1k output, of which 260–1,034 were reasoning tokens.
  - The event stream does not echo model or effort. Both are set only on the command line (`-m gpt-6-astra`, `model_reasoning_effort="high"`).

## FS-3
- `scripts/fs3_reader.py`. Phi-4 @2db69c1c (pinned), greedy, bf16, sdpa, batch 8 left-padded, max_new_tokens 700. Covers 20 captions x 6 conditions = 120 inputs. nice -n 10, own page cache evicted before load. 3,008 s.
  - Prompt template sha256 (head + question block) db82baad13cc1a72219d1ef5009f73c93f8da0cdf98738679dbe2da5cf2ca632. Each prompt's sha256 is in reader/answers.jsonl. full_json = json.dumps(caption, indent=1).
- FS-E1 (bug, parser): the first parse returned 0/120 valid. Phi-4 writes `"id": 1` (integer, no "Q"), and my regex also required the "Q" form. It then adds a prose explanation after the JSON.
  - Fix: `scripts/reader_parse.py` takes the first decodable JSON object and maps 5 -> "Q5". Re-parsed from the saved raw outputs (`fs3_reparse.py`); no model call. 103/120 valid.
- FS-E2 (bug, my token limit): 3 full_json items (main 0bf20967, main 9931d1e8, retest 0bf20967) reasoned in prose and hit 700 tokens before any JSON. My limit is not part of the design.
  - Rerun of those 3 alone, unbatched, with max_new_tokens 2000 (`fs3_rerun_truncated.py`; prompt hashes asserted equal). All 3 valid. The 700-token outputs are kept as raw_output_700.
  - Note: 2 of the 3 new outputs do not start with the same text as the batched run. Batched left-padded bf16 and unbatched greedy decoding differ numerically, so these 3 items were not decoded under the exact same numeric conditions as the other 117.
- Final: 106/120 outputs have 16 yes/no answers. 14 contain "unknown" for some questions despite the best-guess instruction. Unknown answers per condition (of 320): w25 15, w50 11, w100 19, w200 7, w400 2, full_json 35. They count as wrong, as written in fs4_readouts.py.

## FS-4
- `scripts/fs4_readouts.py` -> readouts.json, readouts_tables.md. Classification uses the main caption's reader answers; the retest caption feeds only the test-retest readout.
- REPORT_FS_PILOT.md written. Pilot stopped.
