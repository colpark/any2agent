"""QC-2 Astra practice answers. Usage: qc2.py check | main | soft
Practice order: seed 0 permutation, 60 groups of 5. Half A = first 60 practice semantic questions, half B = second 60.
check: group 1, both halves, given and reversed question order (4 calls; the given-order calls are main calls). Stops below 0.80 agreement.
main:  the other 118 calls, concurrency 3, one retry for invalid or voided calls. soft: soft labels -> ~/langres_state/qc/soft_practice.npy (300 x 120)."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jsonschema
from qc_common import *
PROMPT = ("You see five Sentinel-2 L2A patches, P1 to P5. Each is 1.2 km on a side with north up and has two images in order: first true color "
          "(B04, B03, B02), then false color (B08, B04, B03), where near infrared appears red. Images 1 and 2 show P1, images 3 and 4 show P2, and so on. "
          "Answer from the images only. Do not run commands or use tools. For each patch and each question below, answer yes or no with a confidence "
          "of low, medium or high, using the definition to settle borderline cases. Judge each patch on its own. {questions}")
PIDS = ["P1", "P2", "P3", "P4", "P5"]
SQ = [q for q in questions() if q["split"] == "practice" and q["kind"] == "semantic"]; HALF = {"A": SQ[:60], "B": SQ[60:120]}
qnum = lambda q: int(q["id"][1:])


def schema(half):
    ids = sorted(qnum(q) for q in HALF[half])
    A = {"type": "object", "additionalProperties": False, "required": ["id", "answer", "confidence"], "properties": {"id": {"type": "integer", "enum": ids},
         "answer": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}
    return {"type": "object", "additionalProperties": False, "required": ["patches"], "properties": {"patches": {"type": "array", "minItems": 5, "maxItems": 5, "items": {
        "type": "object", "additionalProperties": False, "required": ["pid", "answers"], "properties": {"pid": {"type": "string", "enum": PIDS},
        "answers": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": A}}}}}}


def validator(half):
    S = schema(half); ids = sorted(qnum(q) for q in HALF[half])
    def v(o):
        e = [x.message for x in jsonschema.Draft202012Validator(S).iter_errors(o)]
        if not e:
            if sorted(p["pid"] for p in o["patches"]) != PIDS: e.append("pids != P1..P5")
            e += [f"{p['pid']} ids != half" for p in o["patches"] if sorted(a["id"] for a in p["answers"]) != ids]
        return e
    return v


def groups():
    order = np.random.default_rng(0).permutation(rows()["practice"]); return [order[k:k + 5] for k in range(0, 300, 5)]


def one(g, half, rev=False, imgs=None):
    qs = HALF[half][::-1] if rev else HALF[half]; block = "\n".join(f"{qnum(q)}. {q['question']} Definition: {q['definition']}" for q in qs)
    tag = f"g{g:02d}_{half}{'_rev' if rev else ''}"; G = groups()[g]
    for attempt in (1, 2):
        rec, obj = call(tag + (f"_try{attempt}" if attempt > 1 else ""), PROMPT.replace("{questions}", block), f"{ST}/schema_{half}.json",
                        [p for i in G for p in imgs[int(i)]], validator(half))
        if rec is None or rec["status"] == "usage_limit": return None
        if rec["status"] == "ok":
            res = {"group": g, "half": half, "rev": rev, "call_id": rec["call_id"], "patch_rows": [int(i) for i in G],
                   "answers": {int(i): {a["id"]: [a["answer"], a["confidence"]] for a in p["answers"]} for i, p in zip(G, sorted(obj["patches"], key=lambda p: p["pid"]))}}
            open(f"{QC}/answers.jsonl", "a").write(json.dumps(res) + "\n"); return res
    return None


def run(jobs):
    for h in HALF: json.dump(schema(h), open(f"{ST}/schema_{h}.json", "w"), indent=1)
    thc.ST = ST; imgs = thc.render("rlpool", sorted({int(i) for g, _, _ in jobs for i in groups()[g]}))  # stage 1 renders, under ~/langres_state/qc/renders
    with ThreadPoolExecutor(3) as ex: out = list(ex.map(lambda j: one(*j, imgs=imgs), jobs))
    print("calls used", used(), "/", CAP, "usage_limit", STOP.is_set()); return out


def done():
    return {(r["group"], r["half"], r["rev"]) for r in map(json.loads, open(f"{QC}/answers.jsonl"))} if os.path.exists(f"{QC}/answers.jsonl") else set()


def check():
    run([(0, h, r) for h in "AB" for r in (False, True) if (0, h, r) not in done()])
    R = {(r["half"], r["rev"]): r["answers"] for r in map(json.loads, open(f"{QC}/answers.jsonl")) if r["group"] == 0}
    ag = [R[(h, False)][i][q][0] == R[(h, True)][i][q][0] for h in "AB" for i in R[(h, False)] for q in R[(h, False)][i]]
    res = {"n_compared": len(ag), "agreement": float(np.mean(ag)), "pass": len(ag) == 600 and float(np.mean(ag)) >= .80}
    json.dump(res, open(f"{QC}/order_check.json", "w"), indent=1); print("ORDER_CHECK", res)


def main():
    assert json.load(open(f"{QC}/order_check.json"))["pass"]
    run([(g, h, False) for g in range(60) for h in "AB" if (g, h, False) not in done()])


def soft():
    pr = {int(i): k for k, i in enumerate(rows()["practice"])}; P = np.full((300, 120), np.nan); col = {qnum(q): j for j, q in enumerate(SQ[:120])}
    for r in map(json.loads, open(f"{QC}/answers.jsonl")):
        if r["rev"]: continue
        for i, A in r["answers"].items():
            for q, (a, c) in A.items(): P[pr[int(i)], col[int(q)]] = SOFT[(a, c)]
    np.save(f"{ST}/soft_practice.npy", P); print("missing", int(np.isnan(P).sum()), "yes rate range", np.round(np.nanmean(P >= .5, 0)[[0, -1]], 3),
                                               "share yes <10%", float((np.nanmean(P >= .5, 0) < .1).mean()))


if __name__ == "__main__":
    {"check": check, "main": main, "soft": soft}[sys.argv[1]]()
