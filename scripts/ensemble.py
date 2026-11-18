"""LR-X26 prior-ensemble pilot: frozen VLM writers -> Phi-4 fusion -> frozen Qwen s and InternVL s'.
Stages:
  ensemble.py write --writer LOV --split val --out DIR     # greedy + 5 samples per patch -> DIR/writer_LOV.jsonl
  ensemble.py fuse  --split val --out DIR --ref_run RUN    # Phi-4 builds every condition -> DIR/{cond}_final.jsonl
  ensemble.py score --split val --out DIR --ref_run RUN    # Qwen s (+u_txt) then InternVL s' (+_final_check) + summary
Writers never score; the scorer and the check model never write (except the E1_QWEN reference, flagged).
"""
import argparse, json, os, sys, glob, itertools, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *

WRITERS = {"LOV": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "PIX": "mistral-community/pixtral-12b",
           "IDF": "HuggingFaceM4/Idefics3-8B-Llama3"}
FUSER = "microsoft/phi-4"
N_SAMP = 5

P_ONE = ("Here is a description of a Sentinel-2 satellite image patch:\n\n{d}\n\n"
         "Rewrite it as one description of at most 100 words for a scientist. Keep every concrete detail about land cover, "
         "vegetation, water, built structures, texture and spatial layout. Do not add anything that is not stated. "
         "Output only the description.")
P_UNION = ("Here are {k} descriptions of the same Sentinel-2 satellite image patch, written independently:\n\n{ds}\n\n"
           "Write one description of at most 100 words for a scientist that combines them. Include every concrete detail "
           "stated by any description unless another description contradicts it; when descriptions contradict each other, "
           "leave that detail out. Do not add anything that is not stated. Output only the description.")
P_CONS = ("Here are {k} descriptions of the same Sentinel-2 satellite image patch, written independently:\n\n{ds}\n\n"
          "Write one description of at most 100 words for a scientist that includes only the details that at least two of "
          "the descriptions state. Do not add anything that is not stated. Output only the description.")


def rows(p):
    return [json.loads(l) for l in open(p)]


def dump(p, R):
    with open(p, "w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")


def listing(ds):
    return "\n\n".join(f"Description {i + 1}: {d}" for i, d in enumerate(ds))


# ---------------------------------------------------------------- write
def write(a):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    repo = WRITERS[a.writer]
    proc = AutoProcessor.from_pretrained(repo, revision=rev(repo))
    tok = proc.tokenizer; tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev(repo), dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to("cuda").eval()
    evict_model_cache(repo)
    pkw = {"do_image_splitting": False} if a.writer == "IDF" else {}  # LR-X26: 120 px source upsampled to 448; tiles add nothing
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DESCRIBE}]}]
    prompt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    pids = ids(a.split); rgb = np.load(f"{ARR}/{a.split}_rgb.npy", mmap_mode="r")
    N = min(len(pids), a.limit or len(pids))
    out_p = f"{a.out}/writer_{a.writer}.jsonl"; R = []; t0 = time.time()
    for k in range(0, N, a.bs):
        idx = list(range(k, min(N, k + a.bs)))
        imgs = [to_img(rgb[i]) for i in idx]
        enc = proc(text=[prompt] * len(idx), images=[[im] for im in imgs], padding=True, return_tensors="pt", **pkw).to("cuda")
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=160, do_sample=False)
            torch.manual_seed(a.seed + k)
            s = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               num_return_sequences=N_SAMP)
        G = [t.strip() for t in tok.batch_decode(g[:, L:], skip_special_tokens=True)]
        S = [t.strip() for t in tok.batch_decode(s[:, L:], skip_special_tokens=True)]
        for j, i in enumerate(idx):
            R.append({"patch_id": pids[i], "writer": a.writer, "repo": repo, "greedy": G[j], "samples": S[j * N_SAMP:(j + 1) * N_SAMP]})
        dump(out_p, R)
        print(a.writer, len(R), "/", N, round(time.time() - t0), "s", flush=True)
    print("WRITE_DONE", a.writer, round(time.time() - t0))


# ---------------------------------------------------------------- fuse
def fuse(a):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    W = {w: rows(f"{a.out}/writer_{w}.jsonl") for w in WRITERS if os.path.exists(f"{a.out}/writer_{w}.jsonl")}
    names = sorted(W); N = min(len(v) for v in W.values())
    pids = ids(a.split)[:N]
    for w in names:
        assert [r["patch_id"] for r in W[w][:N]] == pids
    qwen = rows(f"{a.ref_run}/B2_zeroshot_final.jsonl")[:N]
    jobs = []  # (cond, i, prompt, meta)
    for i in range(N):
        g = {w: W[w][i]["greedy"] for w in names}
        for w in names:
            jobs.append((f"E1_{w}", i, P_ONE.format(d=g[w])))
            jobs.append((f"E2_{w}", i, P_UNION.format(k=len(names), ds=listing(W[w][i]["samples"][:len(names)]))))  # K matched to E3_all
        jobs.append(("E1_QWEN", i, P_ONE.format(d=qwen[i]["summary"])))
        for r in range(2, len(names) + 1):
            for sub in itertools.combinations(names, r):
                jobs.append((f"E3_{'+'.join(sub)}", i, P_UNION.format(k=r, ds=listing([g[w] for w in sub]))))
        jobs.append(("E4_all", i, P_CONS.format(k=len(names), ds=listing([g[w] for w in names]))))
    tok = AutoTokenizer.from_pretrained(FUSER, revision=rev(FUSER)); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(FUSER, revision=rev(FUSER), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    evict_model_cache(FUSER)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for _, _, p in jobs]
    order = np.argsort([len(t) for t in texts]); outs = [None] * len(jobs); t0 = time.time()
    for k in range(0, len(jobs), a.bs):
        b = order[k:k + a.bs]
        enc = tok([texts[j] for j in b], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=200, do_sample=False)
        for j, t in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[j] = t.strip()
        print("fuse", min(k + a.bs, len(jobs)), "/", len(jobs), round(time.time() - t0), "s", flush=True)
    C = {}
    for (c, i, p), t in zip(jobs, outs):
        C.setdefault(c, [None] * N)[i] = t
    for c, ws in C.items():
        dump(f"{a.out}/{c}_final.jsonl", [{"patch_id": pids[i], "arm": "A", "condition": c, "split": a.split, "summary": ws[i],
                                           "n_words": n_words(ws[i]), "fusion_input": [x for x in jobs if x[0] == c and x[1] == i][0][2]}
                                          for i in range(N)])
    dump(f"{a.out}/E5_concat_final.jsonl", [{"patch_id": pids[i], "arm": "A", "condition": "E5_concat", "split": a.split,
                                             "summary": " ".join(W[w][i]["greedy"] for w in names),
                                             "n_words": n_words(" ".join(W[w][i]["greedy"] for w in names))} for i in range(N)])
    print("FUSE_DONE", round(time.time() - t0))


# ---------------------------------------------------------------- score
def score(a):
    conds = sorted(os.path.basename(p)[:-len("_final.jsonl")] for p in glob.glob(f"{a.out}/E*_final.jsonl") if "E6_bestof" not in p)
    R = {c: rows(f"{a.out}/{c}_final.jsonl") for c in conds}
    N = len(R[conds[0]]); pids = ids(a.split)[:N]
    ui = np.load(f"{a.ref_run}/u_img.npy")[:N]; uic = np.load(f"{a.ref_run}/check_u_img.npy")[:N]
    _, mu_txt = load_means("A"); cm = np.load(f"{CAL}/check_means_A.npz")
    sc = QwenScorer(); evict_model_cache(VLM)
    S = {}
    for c in conds:
        ut = unit(sc.pass2([r["summary"] for r in R[c]]), mu_txt); np.save(f"{a.out}/{c}_u_txt.npy", ut.astype(np.float32))
        S[c] = np.einsum("nd,nd->n", ui, ut)
    # E6: best E1 writer per patch by s (scorer-selected)
    e1 = [c for c in conds if c.startswith("E1_") and c != "E1_QWEN"]
    best = np.argmax(np.stack([S[c] for c in e1]), 0)
    R["E6_bestof"] = [{**R[e1[b]][i], "condition": "E6_bestof", "picked": e1[b]} for i, b in enumerate(best)]
    S["E6_bestof"] = np.array([S[e1[b]][i] for i, b in enumerate(best)])
    np.save(f"{a.out}/E6_bestof_u_txt.npy", np.stack([np.load(f"{a.out}/{e1[b]}_u_txt.npy")[i] for i, b in enumerate(best)]))
    conds = conds + ["E6_bestof"]
    del sc; torch.cuda.empty_cache()
    from check_scorer import InternScorer
    cs = InternScorer(); SP = {}
    for c in conds:
        if c == "E6_bestof":
            continue
        SP[c] = np.einsum("nd,nd->n", uic, unit(cs.pass2([r["summary"] for r in R[c]]), cm["mu_txt"]))
    SP["E6_bestof"] = np.array([SP[e1[b]][i] for i, b in enumerate(best)])
    for c in conds:
        dump(f"{a.out}/{c}_final.jsonl", [{**r, "s": float(S[c][i])} for i, r in enumerate(R[c])])
        dump(f"{a.out}/{c}_final_check.jsonl", [{"patch_id": r["patch_id"], "s_prime": float(SP[c][i])} for i, r in enumerate(R[c])])
    # references from the existing val run
    for c in ["B2_zeroshot", "A_search"]:
        S[c] = np.array([r["s"] for r in rows(f"{a.ref_run}/{c}_final.jsonl")])[:N]
        SP[c] = np.array([r["s_prime"] for r in rows(f"{a.ref_run}/{c}_final_check.jsonl")])[:N]
        R[c] = rows(f"{a.ref_run}/{c}_final.jsonl")[:N]
    allc = conds + ["B2_zeroshot", "A_search"]
    I = np.random.default_rng(0).integers(0, N, (1000, N))
    ci = lambda x: [float(x.mean()), float(np.percentile(x[I].mean(1), 2.5)), float(np.percentile(x[I].mean(1), 97.5))]
    out = {"n": N, "s": {c: ci(S[c]) for c in allc}, "s_prime": {c: ci(SP[c]) for c in allc},
           "words_mean": {c: float(np.mean([n_words(r["summary"]) for r in R[c]])) for c in allc},
           "frac_over100": {c: float(np.mean([n_words(r["summary"]) > 100 for r in R[c]])) for c in allc},
           "diff_vs_B2": {c: {"s": ci(S[c] - S["B2_zeroshot"]), "s_prime": ci(SP[c] - SP["B2_zeroshot"])} for c in allc},
           "E6_pick_counts": {c: int((best == k).sum()) for k, c in enumerate(e1)}}
    # coverage curve: mean over subsets of each size (size 1 = E1 rewrite, size >= 2 = E3 union)
    cov = {}
    for c in conds:
        if c.startswith("E1_") and c != "E1_QWEN":
            cov.setdefault(1, []).append(c)
        if c.startswith("E3_"):
            cov.setdefault(c.count("+") + 1, []).append(c)
    out["coverage"] = {k: {"s": float(np.mean([S[c].mean() for c in v])), "s_prime": float(np.mean([SP[c].mean() for c in v])), "conds": v}
                       for k, v in sorted(cov.items())}
    json.dump(out, open(f"{a.out}/ensemble_readout_{a.split}.json", "w"), indent=1)
    for c in allc:
        print(f"{c:<22} s {S[c].mean():.3f}  s' {SP[c].mean():.3f}  words {out['words_mean'][c]:.0f}  >100 {out['frac_over100'][c]:.2f}")
    print("coverage", {k: (round(v['s'], 3), round(v['s_prime'], 3)) for k, v in out["coverage"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--writer"); ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True); ap.add_argument("--ref_run"); ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    # GB10 unified memory: page cache from concurrent downloads blocks CUDA context creation (LR-X26 note);
    # drop our own cached HF blobs and data before touching CUDA, then retry once on OOM at load.
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "evict_cache.py"),
                    os.path.expanduser("~/.cache/huggingface/hub"), f"{HOME}/langres_data", f"{HOME}/langres_state/runs"], capture_output=True)
    {"write": write, "fuse": fuse, "score": score}[a.stage](a)
