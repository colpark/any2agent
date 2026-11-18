# CHANGE_REQUEST LR-X26: prior-ensemble pilot (frozen writers + fusion), arm A, val

Status: APPROVED by the user on 2026-11-18 ("install what's necessary, sounds good. go ahead").
This adds a new family of conditions. DESIGN_v1.md is unchanged. The scorer (Qwen2.5-VL-7B, layer 21, frozen means) and the check model (InternVL3-8B) are unchanged and never write.

Writers, all frozen, HF generate, DESCRIBE prompt, RGB image at 448:
- llava-hf/llava-onevision-qwen2-7b-ov-hf (LOV)
- mistral-community/pixtral-12b (PIX)
- HuggingFaceM4/Idefics3-8B-Llama3 (IDF)

Each writer produces 1 greedy caption and 5 samples (T = 1.0, seed 0) per patch, with max 160 new tokens.

Fusion model: microsoft/phi-4 (text only, greedy, max 200 new tokens). It sees no image.

Conditions (Phi-4 prompts in scripts/ensemble.py):
- E1_<w>: one writer's greedy caption, rewritten to at most 100 words in one style (K = 1).
- E2_<w>: the same writer's first 3 samples (K matched to E3 with all 3 writers) fused with "keep everything not contradicted". Control: sample diversity versus model diversity.
- E3_<subset>: greedy captions of a writer subset fused with "keep everything not contradicted". All subsets of size 2 and 3 give the coverage curve.
- E4_all: consensus fusion of all 3 writers, keeping claims stated by at least 2.
- E5_concat: the 3 greedy captions joined, with no limit and no fusion.
- E6_bestof: the best E1 per patch by s (selection; judged on s').
- Reference: E1_QWEN, Qwen B2 greedy val rewritten by Phi-4 with K = 1 (the scorer writes, so it is marked as a reference). Also B2_zeroshot and A_search from runs/val_A.

Readouts: s, s' with 95% paired bootstrap CI, paired differences, words.
Arm A only, val (100 patches). The residual stays r = u_img - u_txt; no soft tokens.
