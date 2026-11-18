"""TH-1 / TH-3 VLM steps on node A2 (vLLM, bf16, greedy, thinking off). One model for the whole stage.
Usage: th_vlm.py --model REPO --rev SHA (--teacher | --captions | --smoke)
--teacher: bank pass 1, bank pass 2 (classes reversed), test pass 1; JSON-schema guided (19 x {answer, confidence}).
--captions: arm B describe captions for Test300, max 220 tokens.  --smoke: load + one two-image prompt (TH-0 check).
Inputs (copied to A2): teacher_heads/{code_teacher.json, test300.json}; ~/langres_state/fb/bank.json. Outputs: ~/langres_state/th/vlm_*.jsonl"""
import argparse, hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.expanduser("~/Documents/langres/scripts"))
import numpy as np
from common import ARR, STATE, REPO, DESCRIBE, image_user, arm_images, n_words


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--rev", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--teacher", "--captions", "--smoke"): g.add_argument(f, action="store_true")
    a = ap.parse_args()
    OUT = f"{STATE}/th"; TH = f"{REPO}/teacher_heads"; os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    global proc, llm
    proc = AutoProcessor.from_pretrained(a.model, revision=a.rev)
    llm = LLM(a.model, revision=a.rev, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 2}, seed=0)
    HEAD = ("You see two renderings of the same Sentinel-2 L2A patch, 1.2 km on each side, north up. Image 1 is true color (B04, B03, B02). "
            "Image 2 is false color (B08, B04, B03): near infrared appears red. For each land cover class below, answer whether any part of the patch "
            "belongs to it, following the CORINE Land Cover nomenclature, with yes or no and a confidence of low, medium or high.")


    def chat(text):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


    def images(split, idx):
        rgb = np.load(f"{ARR}/{split}_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/{split}_fcc.npy", mmap_mode="r")
        return [arm_images("B", rgb[i], fcc[i]) for i in idx]


    def run(prompts, ims, sp):
        t0 = time.time(); o = llm.generate([{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(prompts, ims)], sp)
        return [x.outputs[0] for x in o], round(time.time() - t0)


    meta = {"model": a.model, "revision": a.rev, "vllm_args": "bf16, max_model_len 8192, gpu_mem 0.5, seed 0, greedy, enable_thinking=False"}
    if a.smoke:
        o, sec = run([chat("Describe the two images in one sentence each.")], images("test", [0]), SamplingParams(temperature=0, max_tokens=80))
        print("SMOKE_OK", sec, repr(o[0].text[:300]))
        sys.exit(0)
    if a.teacher:
        S = json.load(open(f"{TH}/sets.json")); defs = {d["id"]: d["definition"] for d in json.load(open(f"{TH}/code_teacher.json"))["definitions"]}
        names = json.load(open(f"{TH}/classes.json")); order = list(range(1, 20))
        text = HEAD + " " + "\n".join(f"{c}. {names[c - 1]['name']}: {defs[c]}" for c in order)
        schema = {"type": "object", "additionalProperties": False, "required": [str(c) for c in order], "properties": {str(c): {
            "type": "object", "additionalProperties": False, "required": ["answer", "confidence"], "properties": {
                "answer": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}} for c in order}}
        sp = SamplingParams(temperature=0, max_tokens=700, seed=0, structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True))
        for tag, split, idx in [("bank", "rlpool", S["bank_rl"]), ("test", "test", list(range(1000)))]:
            p = chat(text); outs, sec = run([p] * len(idx), images(split, idx), sp); bad = 0
            with open(f"{OUT}/qwen_{tag}.jsonl", "w") as f:
                for i, o in zip(idx, outs):
                    try:
                        j = json.loads(o.text); ans = {str(c): j[str(c)]["answer"] for c in order}; conf = {str(c): j[str(c)]["confidence"] for c in order}
                    except Exception:
                        ans, conf = None, None; bad += 1
                    f.write(json.dumps({"split_index": int(i), "answers": ans, "confidence": conf, "raw": o.text, "finish": o.finish_reason}) + "\n")
            meta[tag] = {"n": len(idx), "sec": sec, "unparsed": bad, "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}
            print(tag, meta[tag], flush=True)
        json.dump(meta, open(f"{OUT}/qwen_teacher_meta.json", "w"), indent=1)
    if a.captions:
        idx = json.load(open(f"{TH}/sets.json"))["test300"]
        msg = [image_user("B", DESCRIBE)]
        p = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        outs, sec = run([p] * len(idx), images("test", idx), SamplingParams(temperature=0, max_tokens=220, seed=0))
        with open(f"{OUT}/qwen_captions_test300.jsonl", "w") as f:
            for i, o in zip(idx, outs):
                f.write(json.dumps({"test_index": i, "caption": o.text.strip(), "n_words": n_words(o.text), "finish": o.finish_reason}) + "\n")
        meta["captions"] = {"n": len(idx), "sec": sec, "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(),
                            "finish_length": sum(o.finish_reason == "length" for o in outs)}
        json.dump(meta, open(f"{OUT}/vlm_captions_meta.json", "w"), indent=1); print(meta["captions"])


if __name__ == "__main__":
    main()
