"""FB-4 readouts: accuracy and skill per tag for every answerer, paired sign tests on 170 items, per-question table,
neighbor label check (report only), agent methods and commands. Missing answers count as wrong. Answerers that did not run are skipped."""
import json, os
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; ST = f"{H}/langres_state/fs"; FB = f"{H}/langres_state/fb"
QS = json.load(open(f"{FS}/fb/questions17.json")); QID = [q["id"] for q in QS]
TAG = {q["id"]: ("shortwave" if q["id"] in ("Q11", "Q12", "Q13") else q["tag"]) for q in QS}
sel = json.load(open(f"{ST}/selection.json")); U, TI = sel["uid_order"], sel["test100_index"]
Kt = np.load(f"{ST}/keys_test100.npz")["keys"]; q17 = json.load(open(f"{FS}/fb/q17_facts.json"))
KEY = {u: {**dict(zip(QID[:16], map(bool, Kt[j]))), "Q17": q17["key_selected"][u]} for u, j in zip(U, TI)}
yn = lambda b: "yes" if b else "no"
ANS = {"majority": {u: {q["id"]: q["majority"] for q in QS} for u in U}}
pil = {}
for l in open(f"{FS}/reader/answers.jsonl"):
    r = json.loads(l)
    if r["role"] == "main" and r["cond"] == "w100": pil[r["uid"]] = r["answers"]
fbr = [json.loads(l) for l in open(f"{FS}/fb/reader_answers.jsonl")]
ANS["C0"] = {u: {**pil[u], **[r for r in fbr if r["uid"] == u and r["cond"] == "C0_Q17"][0]["answers"]} for u in U}
for c in ["C1", "C2"]:
    ANS[c] = {u: [r for r in fbr if r["uid"] == u and r["cond"] == c][0]["answers"] for u in U}
for c in ["A0", "A1"]:
    p = f"{FS}/fb/agent_{c}.json"
    if os.path.exists(p):
        ANS[c] = {t["target_id"]: {e["id"]: e["answer"] for e in t["entries"]} for t in json.load(open(p))["answers"]}
ANS["astra_images"] = {u: {a["id"]: a["answer"] for a in json.load(open(f"{FS}/raw/qa_main_{u}.json"))["answers"]} for u in U}
probes = {p["id"]: p for p in json.load(open(f"{FS}/probes.json"))}
ANS["probe"] = {u: {**{q: yn(probes[q]["pred_selected"][u]) for q in QID[:16]}, "Q17": yn(q17["probe_pred_selected"][u])} for u in U}

R = {a: {(u, q): ANS[a].get(u, {}).get(q) == yn(KEY[u][q]) for u in U for q in QID} for a in ANS}
TAGS = ["visible", "shortwave", "texture", "q17", "all17", "all16"]
inT = lambda q, t: t == "all17" or (t == "all16" and q != "Q17") or TAG[q] == t
acc = {a: {t: float(np.mean([R[a][(u, q)] for u in U for q in QID if inT(q, t)])) for t in TAGS} for a in R}
for a in acc:
    if a == "astra_images": acc[a]["q17"] = None; acc[a]["all17"] = None
skill = {a: {t: (None if acc[a][t] is None else acc[a][t] - acc["majority"][t]) for t in TAGS} for a in acc}


def sign(a, b, qset=None):
    items = [(u, q) for u in U for q in QID if qset is None or q in qset]
    w = sum(R[a][i] and not R[b][i] for i in items); l = sum(R[b][i] and not R[a][i] for i in items)
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return {"n_items": len(items), f"{a}_right_{b}_wrong": w, f"{b}_right_{a}_wrong": l, "acc_diff": (w - l) / len(items), "p_two_sided": round(float(p), 4)}


con = {}
for a, b in [("C1", "C2"), ("C1", "C0"), ("A1", "A0"), ("A1", "probe")]:
    if a in R and b in R: con[f"{a}-{b}"] = sign(a, b)
# neighbor label check (report only)
desc = pd.read_parquet(f"{H}/langres_data/arrays/descriptors.parquet").set_index("patch_id")
bank = json.load(open(f"{FB}/bank.json"))["patch_ids"]; NB = json.load(open(f"{FB}/neighbors.json"))
t100 = json.load(open(f"{H}/langres_data/arrays/test100_ids.json"))
nbchk = {}
for u, j in zip(U, TI):
    tl = set(desc.loc[t100[j], "labels"])
    nbchk[u] = {k: float(np.mean([bool(tl & set(desc.loc[bank[b], "labels"])) for b, _ in NB[u][k]])) for k in ["neighbors", "random"]}
nb_mean = {k: float(np.mean([v[k] for v in nbchk.values()])) for k in ["neighbors", "random"]}
perq = {q: {"key_yes": int(sum(KEY[u][q] for u in U)), **{a: int(sum(R[a][(u, q)] for u in U)) for a in R}} for q in QID}
fbv = Counter(r["valid"] for r in fbr)
out = {"accuracy": acc, "skill": skill, "contrasts": con, "per_question": perq, "neighbor_label_share": nbchk, "neighbor_label_share_mean": nb_mean,
       "fb_reader_valid": dict(fbv), "answerers": list(R)}
json.dump(out, open(f"{FS}/fb/readouts.json", "w"), indent=1)
f = lambda x: "—" if x is None else f"{x:.2f}"
fs = lambda x: "—" if x is None else f"{x:+.2f}"
L = ["| Answerer | " + " | ".join(TAGS) + " |", "|---|" + "---|" * len(TAGS)]
for a in R: L.append(f"| {a} | " + " | ".join(f"{f(acc[a][t])} ({fs(skill[a][t])})" for t in TAGS) + " |")
L += ["", "| Q | Tag | Key yes | " + " | ".join(R) + " |", "|---|---|---|" + "---|" * len(R)]
for q in QID: L.append(f"| {q} | {TAG[q]} | {perq[q]['key_yes']} | " + " | ".join(str(perq[q][a]) for a in R) + " |")
open(f"{FS}/fb/readouts_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(json.dumps({"contrasts": con, "neighbor_label_share_mean": nb_mean, "fb_reader_valid": dict(fbv)}, indent=1))
