"""FN-3. `run` (node A2): Qwen3.5-9B (pinned), vLLM bf16, greedy, thinking off, guided {"answer": "1"|"2"}; D1 and D0s on the
same 800 pairs. `grade` (host A): measures, verdict N1, figure. Composites are copied to A2 at the same paths."""
import json, os, sys, time
import numpy as np
H = os.path.expanduser("~"); FN = f"{H}/Documents/langres/fm_names"; ST = f"{H}/langres_state/fn"
QWEN, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
P = ("You see two Sentinel-2 L2A patches, each 1.2 km on a side with north up. Each image shows one patch, with true color (B04, B03, B02) on the "
     "left and false color (B08, B04, B03), where near infrared appears red, on the right. Another model describes a contrast between two ends. "
     "End A: {name_a}. End B: {name_b}. Definition: {definition} Cues to check: {cues}. Which patch lies closer to end A, image 1 or image 2?")


def derangement(n=20, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any(): return p


def run():
    import hashlib, subprocess
    from PIL import Image
    subprocess.run([sys.executable, f"{H}/Documents/langres/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    F1, N = json.load(open(f"{FN}/fn1.json")), json.load(open(f"{FN}/names.json")); der = derangement()
    proc = AutoProcessor.from_pretrained(QWEN, revision=REV)
    llm = LLM(QWEN, revision=REV, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 2}, seed=0)
    jobs = []
    for d in range(20):
        for cond, src in (("D1", d), ("D0s", int(der[d]))):
            n = N[str(src)]; text = P.format(name_a=n["name_a"], name_b=n["name_b"], definition=n["definition"], cues=" | ".join(n["cues"]))
            prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for k, q in enumerate(F1["pairs"][str(d)]):
                ims = [q["img_a"], q["img_b"]] if q["a_pos"] == 1 else [q["img_b"], q["img_a"]]
                jobs.append({"d": d, "cond": cond, "desc_from": src, "k": k, "a_pos": q["a_pos"], "prompt": prompt, "ims": ims})
    sp = SamplingParams(temperature=0, max_tokens=20, seed=0, structured_outputs=StructuredOutputsParams(
        json={"type": "object", "additionalProperties": False, "required": ["answer"], "properties": {"answer": {"type": "string", "enum": ["1", "2"]}}}))
    t0 = time.time()
    outs = llm.generate([{"prompt": j["prompt"], "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in j["ims"]]}} for j in jobs], sp)
    with open(f"{ST}/reader_answers.jsonl", "w") as f:
        for j, o in zip(jobs, outs):
            try: a = json.loads(o.outputs[0].text)["answer"]
            except Exception: a = None
            f.write(json.dumps({k: j[k] for k in ("d", "cond", "desc_from", "k", "a_pos")} | {"answer": a, "raw": o.outputs[0].text,
                                "prompt_sha256": hashlib.sha256(j["prompt"].encode()).hexdigest()}) + "\n")
    print("DONE", len(jobs), round(time.time() - t0), "s", "unparsed", sum(1 for o in outs if not o.outputs[0].text.strip()))


def grade():
    sys.path.insert(0, os.path.dirname(__file__))
    from fn_common import wilson
    from scipy.stats import wilcoxon, spearmanr
    F1, N = json.load(open(f"{FN}/fn1.json")), json.load(open(f"{FN}/names.json")); R = [json.loads(l) for l in open(f"{FN}/reader_answers.jsonl")]
    per = {}
    for d in range(20):
        r1 = [r for r in R if r["d"] == d and r["cond"] == "D1"]; r0 = [r for r in R if r["d"] == d and r["cond"] == "D0s"]
        k1 = sum(r["answer"] == str(r["a_pos"]) for r in r1); a0 = np.mean([r["answer"] == str(r["a_pos"]) for r in r0])
        per[d] = {"d1": k1 / len(r1), "d1_ci": wilson(k1, len(r1)), "d0s_raw": float(a0), "d0s": float(max(a0, 1 - a0)), "d0s_from": r0[0]["desc_from"],
                  "named": wilson(k1, len(r1))[0] > 0.5, "evr": F1["evr"][d], **F1["diag"][str(d)], "visible": N[str(d)]["visible"],
                  "name_a": N[str(d)]["name_a"], "name_b": N[str(d)]["name_b"]}
    d1 = np.array([per[d]["d1"] for d in range(20)]); d0 = np.array([per[d]["d0s"] for d in range(20)]); named = [d for d in range(20) if per[d]["named"]]
    w = wilcoxon(d1, d0); vis = {"low": 0, "medium": 1, "high": 2}
    sp = lambda x: [float(v) for v in spearmanr(d1, x)]
    out = {"per_direction": per, "n_named": len(named), "named": named, "mean_d1": float(d1.mean()), "mean_d0s": float(d0.mean()),
           "wilcoxon": {"stat": float(w.statistic), "p": float(w.pvalue)}, "named_variance_share": float(sum(F1["evr"][d] for d in named) / sum(F1["evr"])),
           "spearman_d1": {"pc_rank": sp(np.arange(1, 21)), "swir_gain": sp([per[d]["swir_gain"] for d in range(20)]), "visible": sp([vis[per[d]["visible"]] for d in range(20)])},
           "reader_share_1": {c: float(np.mean([r["answer"] == "1" for r in R if r["cond"] == c])) for c in ("D1", "D0s")},
           "unparsed": sum(r["answer"] is None for r in R)}
    out["verdict_N1"] = out["n_named"] >= 10 and out["mean_d1"] > out["mean_d0s"] and out["wilcoxon"]["p"] < 0.05
    json.dump(out, open(f"{FN}/fn3_measures.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.axhline(0.5, color="#52514e", lw=1, ls="--"); g = [per[d]["swir_gain"] for d in range(20)]
    sc = ax.scatter(np.arange(1, 21), d1, c=g, cmap="Blues", vmin=0, vmax=max(g), s=60, edgecolor="#52514e", linewidth=0.6, zorder=3)
    ax.errorbar(np.arange(1, 21), d1, yerr=[d1 - [per[d]["d1_ci"][0] for d in range(20)], [per[d]["d1_ci"][1] for d in range(20)] - d1], fmt="none", ecolor="#c3c2b7", lw=1, zorder=2)
    ax.set_xlabel("PC rank", color="#52514e"); ax.set_ylabel("D1 accuracy (40 pairs, Wilson 95% CI)", color="#52514e"); ax.set_xticks(range(1, 21)); ax.set_ylim(0, 1.02)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.colorbar(sc, ax=ax, label="SWIR gain (full R² − visible R²)"); ax.set_title("Can Qwen use Astra's name to place patches on each Prithvi direction?", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(f"{FN}/figs/d1_vs_rank.png", dpi=150)
    print(json.dumps({k: v for k, v in out.items() if k != "per_direction"}, indent=1))
    for d in range(20): print(d + 1, "%.2f [%.2f,%.2f]" % (per[d]["d1"], *per[d]["d1_ci"]), "D0s %.2f" % per[d]["d0s"], per[d]["visible"], "swir %.3f" % per[d]["swir_gain"], per[d]["name_a"], "|", per[d]["name_b"])


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()
