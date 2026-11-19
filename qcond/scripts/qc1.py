"""QC-1. Usage: qc1.py ask | keys | embed | build
ask:   1 Astra call writing 150 practice questions (image-QA settings, no images, empty folder) -> qcond/raw, generated.json
keys:  exact keys for the 31 physical questions on Practice and Test -> ~/langres_state/qc/phys_{practice,test}.npy, physical.json
embed: Qwen3-Embedding-4B (VS revision, no prefix, L2-normalized) on every "{question} {definition}" -> emb_questions.npy (GPU)
build: leakage filter (cosine > 0.85 to any held-out question) and questions.json (practice / held out)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jsonschema
from qc_common import *
ASK = ("Write 150 yes or no questions that a scientist might ask about a single Sentinel-2 L2A patch, 1.2 km on a side, seen in a true color image "
       "(B04, B03, B02) and a false color image (B08, B04, B03). Cover land use and land cover, spatial layout, vegetation condition and season, "
       "human activity, water and wetness, and combinations of these. Each question must be answerable by looking at the two images, and roughly "
       "10 to 60 percent of typical European patches should get a yes. Do not ask about location, date or anything outside the images. Do not "
       "simply restate a standard land-cover class name as a question. Give each question a one-sentence definition that settles borderline cases.")
ASK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["questions"], "properties": {"questions": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "question", "definition"],
    "properties": {"id": {"type": "integer"}, "question": {"type": "string"}, "definition": {"type": "string"}}}}}}
D10 = [("brightness", "mean over pixels of (B02 + B03 + B04) / 3"), ("greenness (NDVI)", "mean NDVI, (B08 - B04) / (B08 + B04)"),
       ("variation in greenness", "standard deviation over pixels of NDVI"), ("vegetated share", "share of pixels with NDVI above 0.5"),
       ("water index (NDWI)", "mean NDWI, (B03 - B08) / (B03 + B08)"), ("water share", "share of pixels with NDWI above 0"),
       ("moisture (NDMI)", "mean NDMI, (B8A - B11) / (B8A + B11)"), ("shortwave infrared 1 brightness", "mean of B11"),
       ("shortwave infrared 2 brightness", "mean of B12"), ("near-infrared texture", "standard deviation over pixels of B08")]
P2M = [("greener", "NDVI"), ("higher in water index", "NDWI"), ("moister", "NDMI"), ("brighter", "brightness, (B02 + B03 + B04) / 3,")]
HELD_P = {"P1_6_top", "P1_6_bottom", "P1_8_top", "P1_8_bottom", "P1_9_top", "P1_9_bottom", "P2_moister_lr", "P2_moister_tb", "P3_ndmi", "P3_b12"}
BI = {b: i for i, b in enumerate(["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"])}
nd = lambda a, b: (a - b) / (a + b + 1e-6)


def ask():
    p = f"{ST}/schema_ask.json"; json.dump(ASK_SCHEMA, open(p, "w"), indent=1)
    rec, obj = call("ask", ASK, p, [], lambda o: [e.message for e in jsonschema.Draft202012Validator(ASK_SCHEMA).iter_errors(o)],
                    extra=("--enable", "skip_host_skill_discovery"))  # QC-X1
    if rec and rec["status"] == "ok": json.dump(obj["questions"], open(f"{QC}/generated.json", "w"), indent=1); print(len(obj["questions"]), "questions")


def phys_defs():
    Q = []
    for j, (nm, f) in enumerate(D10):
        for side, pc in (("top", "above the 75th"), ("bottom", "below the 25th")):
            Q.append({"id": f"P1_{j}_{side}", "question": f"Is the patch's {nm} in the {side} quarter of patches?",
                      "definition": f"Yes if the patch's {f} is {pc} percentile of a reference set of 497 Sentinel-2 patches."})
    for word, m in P2M:
        for tag, a, b in (("lr", "left", "right"), ("tb", "top", "bottom")):
            Q.append({"id": f"P2_{word.split()[-1]}_{tag}", "question": f"Is the {a} half {word} than the {b} half?",
                      "definition": f"Yes if the mean {m} over the {a} half of the patch exceeds the mean over the {b} half."})
    vnir = "a linear regression on the B02, B03, B04 and B08 means, NDVI mean and NDWI mean fitted over a reference pool"
    Q.append({"id": "P3_ndmi", "question": "Is the patch moister than its visible and near-infrared appearance predicts?",
              "definition": f"Yes if the patch's mean NDMI, (B8A - B11) / (B8A + B11), exceeds the value predicted by {vnir}."})
    for k, b in (("1", "B11"), ("2", "B12")):
        Q.append({"id": f"P3_{b.lower()}", "question": f"Is the patch brighter in shortwave infrared {k} than its visible and near-infrared appearance predicts?",
                  "definition": f"Yes if the patch's mean {b} exceeds the value predicted by {vnir}."})
    for q in Q: q.update(kind="physical", split="held_out" if q["id"] in HELD_P else "practice")
    return Q


def stats(split, idx):
    """Per patch: B02, B03, B04, B08 means, NDVI mean, NDWI mean | NDMI, B11, B12 means | left-right and top-bottom half differences of NDVI, NDWI, NDMI, brightness."""
    X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r"); out = []
    for k in range(0, len(idx), 500):
        x = np.asarray(X[idx[k:k + 500]], np.float32); g = lambda b: x[:, BI[b]]
        M = {"NDVI": nd(g("B08"), g("B04")), "NDWI": nd(g("B03"), g("B08")), "NDMI": nd(g("B8A"), g("B11")), "bri": (g("B02") + g("B03") + g("B04")) / 3}
        h = [M[m][:, :, :60].mean((1, 2)) - M[m][:, :, 60:].mean((1, 2)) for m in M] + [M[m][:, :60].mean((1, 2)) - M[m][:, 60:].mean((1, 2)) for m in M]
        out.append(np.column_stack([g(b).mean((1, 2)) for b in ("B02", "B03", "B04", "B08")] + [M["NDVI"].mean((1, 2)), M["NDWI"].mean((1, 2)),
                                    M["NDMI"].mean((1, 2)), g("B11").mean((1, 2)), g("B12").mean((1, 2))] + h))
    return np.concatenate(out)


def keys():
    from sklearn.linear_model import LinearRegression
    R = rows(); Q = phys_defs(); Db = np.load(f"{ah.ST}/desc_bank.npy")[ah.bank_ok()]; lo, hi = np.percentile(Db, 25, 0), np.percentile(Db, 75, 0)
    rl = stats("rlpool", np.arange(len(json.load(open(f"{ARR}/rlpool_ids.json"))))); S = {"practice": stats("rlpool", R["practice"]), "test": stats("test", R["test"])}
    D = {"practice": np.load(f"{ah.ST}/desc_bank.npy")[ah.bank_ok()][practice()], "test": np.load(f"{ah.ST}/desc_test.npy")}
    regs = {t: LinearRegression().fit(rl[:, :6], rl[:, c]) for t, c in (("ndmi", 6), ("b11", 7), ("b12", 8))}
    q17 = np.load(f"{H}/langres_state/fb/q17.npz"); facts = {"r2": {t: float(r.score(rl[:, :6], rl[:, c])) for (t, r), c in zip(regs.items(), (6, 7, 8))}}
    facts["q17_matches_fb_rlpool"] = bool(np.array_equal((rl[:, 6] - regs["ndmi"].predict(rl[:, :6])) > 0, q17["y_rlpool"]))
    pm = {"greener": 0, "index": 1, "moister": 2, "brighter": 3}
    for tag in ("practice", "test"):
        s, d, K = S[tag], D[tag], []
        for q in Q:
            p = q["id"].split("_")
            if p[0] == "P1": j = int(p[1]); K.append(d[:, j] > hi[j] if p[2] == "top" else d[:, j] < lo[j])
            elif p[0] == "P2": K.append(s[:, 9 + pm[p[1]] + (4 if p[2] == "tb" else 0)] > 0)
            else: c = {"ndmi": 6, "b11": 7, "b12": 8}[p[1]]; K.append(s[:, c] - regs[p[1]].predict(s[:, :6]) > 0)
        K = np.column_stack(K); np.save(f"{ST}/phys_{tag}.npy", K); facts[f"yes_rate_{tag}"] = dict(zip([q["id"] for q in Q], np.round(K.mean(0), 3).tolist()))
    facts["desc_check_practice_maxdiff"] = float(np.abs(S["practice"][:, 6] - D["practice"][:, 6]).max())
    json.dump({"questions": Q, "facts": facts}, open(f"{QC}/physical.json", "w"), indent=1); print(json.dumps({k: v for k, v in facts.items() if "yes_rate" not in k}))


def heldout_sem():
    from th_common import CLASSES
    C = {c["id"]: c for c in ah.classes()}; names = lambda m: ", ".join(CLASSES[c - 1] for c in m[:-1]) + " or " + CLASSES[m[-1] - 1]
    Q = [{"id": f"C{c}", "question": f"Does any part of the patch belong to {C[c]['name']}, following the CORINE Land Cover nomenclature?",
          "definition": C[c]["definition"]} for c in CL]
    gd = {g: f"Yes if any part of the patch belongs to {names(GM[g])} in the CORINE Land Cover nomenclature." for g in ("G1", "G2", "G3", "G4")}
    gd["G5"] = f"Wetland as defined by its member classes in the CORINE Land Cover nomenclature: {names(GM['G5'])}."
    gd["G6"] = f"Natural open land as defined by its member classes in the CORINE Land Cover nomenclature: {names(GM['G6'])}."
    gd["G7"] = "Yes if the patch contains farmland (any CORINE Land Cover class from arable land to agro-forestry areas) and also forest, wetland or natural open land."
    gd["G8"] = "Yes if the patch contains more than three distinct CORINE Land Cover classes."
    return [{**q, "kind": "semantic", "split": "held_out"} for q in Q] + [{"id": g, "question": t, "definition": gd[g], "kind": "semantic", "split": "held_out"} for g, t in GQ]


def all_texts():
    G = json.load(open(f"{QC}/generated.json"))
    T = heldout_sem() + json.load(open(f"{QC}/physical.json"))["questions"] + [{"id": f"A{q['id']}", "question": q["question"], "definition": q["definition"],
                                                                               "kind": "semantic", "split": "generated"} for q in G]
    return T


def embed():
    import subprocess, time, torch
    from sentence_transformers import SentenceTransformer
    import vs_common
    subprocess.run([thc.PY, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    m = SentenceTransformer(vs_common.EMB, revision=vs_common.EREV, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); t0 = time.time()
    T = all_texts(); E = m.encode([f"{q['question']} {q['definition']}" for q in T], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    np.save(f"{ST}/emb_all.npy", E.astype(np.float32)); json.dump([q["id"] for q in T], open(f"{ST}/emb_all_ids.json", "w")); print(E.shape, round(time.time() - t0, 1), "s")


def build():
    T = all_texts(); E = np.load(f"{ST}/emb_all.npy"); assert json.load(open(f"{ST}/emb_all_ids.json")) == [q["id"] for q in T]
    held = np.array([q["split"] == "held_out" for q in T]); gen = np.array([q["split"] == "generated" for q in T])
    mx = (E[gen] @ E[held].T).max(1); keep = np.flatnonzero(gen)[mx <= .85][:120]
    for k in np.flatnonzero(gen): T[k]["max_cos_heldout"] = float((E[k] @ E[held].T).max())
    for k in keep: T[k]["split"] = "practice"
    out = [dict(q, emb_row=k) for k, q in enumerate(T) if q["split"] != "generated"]
    rep = {"generated": int(gen.sum()), "dropped_leak": [T[k]["id"] for k in np.flatnonzero(gen)[mx > .85]], "kept": len(keep),
           "practice_semantic": [T[k]["id"] for k in keep], "n_practice": sum(q["split"] == "practice" for q in out), "n_held_out": sum(q["split"] == "held_out" for q in out)}
    json.dump(out, open(f"{QC}/questions.json", "w"), indent=1); json.dump(rep, open(f"{QC}/leakage_filter.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "practice_semantic"}))


if __name__ == "__main__":
    {"ask": ask, "keys": keys, "embed": embed, "build": build}[sys.argv[1]]()
