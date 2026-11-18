"""FS-4: classify (patch, question, condition) into said / seen_not_said / not_seen / neither and compute readouts 1-8.
Classification uses the main caption's reader answers; the retest caption feeds the test-retest readout only.
A missing or unparsable reader answer counts as wrong. Outputs frontier_share/readouts.json and readouts_tables.md (tables only);
REPORT_FS_PILOT.md is written by hand from these."""
import json, os, re
from collections import Counter, defaultdict
import numpy as np

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; ST = f"{H}/langres_state/fs"
Qs = json.load(open(f"{FS}/questions.json")); QID = [q["id"] for q in Qs]; TAG = {q["id"]: q["tag"] for q in Qs}
MAJ = {q["id"]: q["majority"] == "yes" for q in Qs}
sel = json.load(open(f"{ST}/selection.json")); U = sel["uid_order"]
Kt = np.load(f"{ST}/keys_test100.npz")["keys"]
KEY = {u: dict(zip(QID, map(bool, Kt[j]))) for u, j in zip(U, sel["test100_index"])}
PROBE = {p["id"]: p for p in json.load(open(f"{FS}/probes.json"))}
CONDS = ["w25", "w50", "w100", "w200", "w400", "full_json"]; TAGS = ["visible", "shortwave", "texture"]
CLS = ["said", "seen_not_said", "not_seen", "neither"]
yn = lambda b: "yes" if b else "no"

QA = {}
for u in U:
    p = f"{FS}/raw/qa_main_{u}.json"
    QA[u] = {a["id"]: a for a in json.load(open(p))["answers"]} if os.path.exists(p) else {}
RD = defaultdict(dict)  # (role, uid) -> cond -> answers
RVALID = Counter()
for line in open(f"{FS}/reader/answers.jsonl"):
    r = json.loads(line); RD[(r["role"], r["uid"])][r["cond"]] = r["answers"]; RVALID[r["valid"]] += 1

right = lambda ans, u, q: ans is not None and ans == yn(KEY[u][q])
rows = []
for u in U:
    for q in QID:
        astra = QA[u].get(q, {}).get("answer"); prob = yn(PROBE[q]["pred_selected"][u])
        for c in CONDS:
            rd = RD[("main", u)].get(c, {}).get(q)
            cl = ("said" if right(rd, u, q) else "seen_not_said" if right(astra, u, q) else "not_seen" if right(prob, u, q) else "neither")
            rows.append({"uid": u, "q": q, "tag": TAG[q], "cond": c, "cls": cl, "reader": rd, "astra": astra, "probe": prob,
                         "key": yn(KEY[u][q]), "astra_conf": QA[u].get(q, {}).get("confidence")})


def table(filter_fn):
    out = {}
    for c in CONDS:
        cnt = Counter(r["cls"] for r in rows if r["cond"] == c and filter_fn(r)); n = sum(cnt.values())
        out[c] = {k: [cnt[k], round(cnt[k] / n, 3) if n else None] for k in CLS} | {"n": n}
    return out


main = {"all": table(lambda r: True)} | {t: table(lambda r, t=t: r["tag"] == t) for t in TAGS}


def acc(fn, tag=None):
    v = [fn(u, q) for u in U for q in QID if tag is None or TAG[q] == tag]
    return round(float(np.mean(v)), 3)


accs = {}
for tag in [None] + TAGS:
    k = tag or "all"
    accs[k] = {"astra_images": acc(lambda u, q: right(QA[u].get(q, {}).get("answer"), u, q), tag),
               "prithvi_probe": acc(lambda u, q: right(yn(PROBE[q]["pred_selected"][u]), u, q), tag),
               "majority_baseline": acc(lambda u, q: MAJ[q] == KEY[u][q], tag)}
    for c in CONDS:
        accs[k][f"reader_{c}"] = acc(lambda u, q, c=c: right(RD[("main", u)].get(c, {}).get(q), u, q), tag)
    accs[k]["reader_retest_w100"] = acc(lambda u, q: right(RD[("retest", u)].get("w100", {}).get(q), u, q), tag)
per_q = {q: {"astra": round(float(np.mean([right(QA[u].get(q, {}).get("answer"), u, q) for u in U])), 2),
             "probe": round(float(np.mean([right(yn(PROBE[q]["pred_selected"][u]), u, q) for u in U])), 2),
             "reader_w100": round(float(np.mean([right(RD[("main", u)].get("w100", {}).get(q), u, q) for u in U])), 2),
             "reader_full_json": round(float(np.mean([right(RD[("main", u)].get("full_json", {}).get(q), u, q) for u in U])), 2),
             "majority": round(float(np.mean([MAJ[q] == KEY[u][q] for u in U])), 2), "key_yes": int(sum(KEY[u][q] for u in U)),
             "probe_cv": round(PROBE[q]["cv_acc"], 3)} for q in QID}

# control check
SW, NDVI = ["Q11", "Q12", "Q13"], ["Q1", "Q2", "Q4"]
ctrl = {}
for c in CONDS + ["pooled"]:
    sub = [r for r in rows if (c == "pooled" or r["cond"] == c)]
    sw = Counter(r["cls"] for r in sub if r["q"] in SW); nv = Counter(r["cls"] for r in sub if r["q"] in NDVI)
    ctrl[c] = {"shortwave": dict(sw), "shortwave_notseen_or_neither": round((sw["not_seen"] + sw["neither"]) / max(1, sum(sw.values())), 3),
               "ndvi": dict(nv), "ndvi_said_or_seen": round((nv["said"] + nv["seen_not_said"]) / max(1, sum(nv.values())), 3)}
p = ctrl["pooled"]
ctrl["verdict"] = {"shortwave_mostly_unseen": p["shortwave_notseen_or_neither"] > 0.5, "ndvi_mostly_seen": p["ndvi_said_or_seen"] > 0.5}


# test-retest
def cap(role, u):
    f = f"{FS}/raw/caption_{role}_{u}.json"
    return json.load(open(f)) if os.path.exists(f) else None


tok = lambda t: set(re.findall(r"[a-z]+", t.lower()))
J = lambda a, b: len(a & b) / max(1, len(a | b))
retest = {}
for c in CONDS:
    ag = [RD[("main", u)][c][q] == RD[("retest", u)][c][q] for u in U for q in QID
          if c in RD[("main", u)] and c in RD[("retest", u)] and RD[("main", u)][c].get(q) is not None]
    jac = [J(tok(cap("main", u)["summaries"][c]), tok(cap("retest", u)["summaries"][c])) for u in U
           if c != "full_json" and cap("main", u) and cap("retest", u)]
    if c == "full_json":
        jac = [J(tok(json.dumps(cap("main", u))), tok(json.dumps(cap("retest", u)))) for u in U if cap("main", u) and cap("retest", u)]
    retest[c] = {"reader_agreement": round(float(np.mean(ag)), 3) if ag else None, "n": len(ag), "word_jaccard": round(float(np.mean(jac)), 3) if jac else None}
cover_tv, top_pct_diff, top_same = [], [], []
for u in U:
    a, b = cap("main", u), cap("retest", u)
    if not (a and b): continue
    pa = sorted([x["percent"] for x in a["global"]["cover"]], reverse=True); pb = sorted([x["percent"] for x in b["global"]["cover"]], reverse=True)
    n = max(len(pa), len(pb)); pa += [0] * (n - len(pa)); pb += [0] * (n - len(pb))
    cover_tv.append(0.5 * sum(abs(x - y) for x, y in zip(pa, pb))); top_pct_diff.append(abs(pa[0] - pb[0]))
    top_same.append(J(tok(a["global"]["cover"][0]["class"]), tok(b["global"]["cover"][0]["class"])))
retest["cover"] = {"sorted_percent_total_variation_mean": round(float(np.mean(cover_tv)), 1), "per_patch_tv": cover_tv,
                   "top_class_percent_absdiff_mean": round(float(np.mean(top_pct_diff)), 1), "top_class_name_word_jaccard_mean": round(float(np.mean(top_same)), 3)}

# caption checks by code
GAZ = ["Europe", "European", "Austria", "Belgium", "Finland", "Ireland", "Kosovo", "Lithuania", "Luxembourg", "Portugal", "Serbia",
       "Switzerland", "Germany", "France", "Spain", "Italy", "Poland", "Sweden", "Norway", "Denmark", "Netherlands", "Estonia", "Latvia",
       "Russia", "Britain", "England", "Scotland", "Balkan", "Balkans", "Baltic", "Scandinavia", "Scandinavian", "Nordic", "Iberia", "Iberian",
       "Alps", "Alpine", "Danube", "Rhine", "Atlantic", "Mediterranean", "Africa", "Asia", "America", "Australia", "Vienna", "Brussels",
       "Helsinki", "Dublin", "Pristina", "Vilnius", "Lisbon", "Belgrade", "Bern", "Zurich", "Geneva", "Porto", "Kaunas", "Tampere",
       "Lapland", "Carpathian", "Pyrenees"]
FAM = {"water": ["water", "lake", "river", "sea", "marine", "reservoir", "pond", "estuar", "lagoon", "channel"],
       "forest": ["forest", "woodland", "tree", "conifer", "broad", "wood", "canopy"],
       "crop": ["arable", "crop", "field", "agricult", "cultivat", "farmland", "plough", "tillage", "orchard", "vineyard", "olive", "fruit"],
       "grass": ["grass", "pasture", "meadow", "herbaceous"],
       "shrub": ["shrub", "scrub", "heath", "moor", "transitional", "sclerophyll"],
       "wet": ["wetland", "bog", "peat", "marsh", "mire", "fen", "intertidal", "salt"],
       "built": ["urban", "built", "settlement", "industrial", "commercial", "road", "building", "residential", "construction", "mine", "dump", "airport", "port", "sport", "leisure"],
       "bare": ["bare", "rock", "beach", "dune", "sand", "sparse", "soil", "burnt", "glacier", "snow"]}
fam = lambda t: {f for f, ws in FAM.items() if any(w in t.lower() for w in ws)}
import pandas as pd
desc = pd.read_parquet(f"{H}/langres_data/arrays/descriptors.parquet").set_index("patch_id")
idmap = {r["uid"]: r["patch_id"] for r in json.load(open(f"{FS}/private/id_map.json"))}
checks = {}
for u in U:
    for role in ["main", "retest"]:
        a = cap(role, u)
        if not a: continue
        cells = [g["cell"] for g in a["grid"]]; text = json.dumps(a, ensure_ascii=False)
        gaz = sorted({g for g in GAZ if re.search(rf"\b{g}\b", text)})
        labels = list(desc.loc[idmap[u], "labels"]); lf = set().union(*[fam(l) for l in labels])
        top = max(a["global"]["cover"], key=lambda x: x["percent"])
        checks[f"{role}:{u[:8]}"] = {"grid_cells_complete": sorted(cells) == sorted(set(cells)) and len(set(cells)) == 9,
                                     "cover_sum": round(sum(x["percent"] for x in a["global"]["cover"]), 1),
                                     "n_features": len(a["features"]), "n_feature_count_sum": sum(f["count"] for f in a["features"]),
                                     "n_interpretation": len(a["interpretation"]), "n_uncertainties": len(a["uncertainties"]),
                                     "summary_words": {k: len(v.split()) for k, v in a["summaries"].items()},
                                     "empty_strings": sum(1 for m in re.finditer(r'": ""', json.dumps(a))),
                                     "gazetteer_hits": gaz, "top_cover": top["class"], "top_cover_percent": top["percent"],
                                     "top_cover_families": sorted(fam(top["class"])), "label_families": sorted(lf),
                                     "top_cover_matches_labels": bool(fam(top["class"]) & lf)}

ledger = [json.loads(l) for l in open(f"{FS}/calls.jsonl")]
out = {"reader_valid": dict(RVALID), "main": main, "accuracy": accs, "per_question": per_q, "control": ctrl, "retest": retest,
       "caption_checks": checks, "calls_used": len(ledger), "ledger_status": dict(Counter(r["status"] for r in ledger)),
       "rows": rows}
json.dump(out, open(f"{FS}/readouts.json", "w"), indent=1)

# markdown tables
L = []
L.append("| Condition | Said | Seen, not said | Not seen | Neither | n | Reader acc | Majority acc |\n|---|---|---|---|---|---|---|---|")
for c in CONDS:
    m = main["all"][c]
    L.append(f"| {c} | " + " | ".join(f"{m[k][0]} ({m[k][1]:.0%})" for k in CLS) + f" | {m['n']} | {accs['all'][f'reader_{c}']:.2f} | {accs['all']['majority_baseline']:.2f} |")
L.append("")
L.append("| Tag | Condition | Said | Seen, not said | Not seen | Neither | n | Majority acc |\n|---|---|---|---|---|---|---|---|")
for t in TAGS:
    for c in CONDS:
        m = main[t][c]
        L.append(f"| {t} | {c} | " + " | ".join(f"{m[k][0]} ({m[k][1]:.0%})" for k in CLS) + f" | {m['n']} | {accs[t]['majority_baseline']:.2f} |")
L.append("")
cols = ["astra_images", "prithvi_probe", "majority_baseline"] + [f"reader_{c}" for c in CONDS]
L.append("| Tag | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
for k in ["all"] + TAGS:
    L.append(f"| {k} | " + " | ".join(f"{accs[k][c]:.2f}" for c in cols) + " |")
L.append("")
L.append("| Q | Tag | Key yes /10 | Astra | Reader w100 | Reader JSON | Probe | Probe CV | Majority |\n|---|---|---|---|---|---|---|---|---|")
for q in QID:
    d = per_q[q]
    L.append(f"| {q} | {TAG[q]} | {d['key_yes']} | {d['astra']:.1f} | {d['reader_w100']:.1f} | {d['reader_full_json']:.1f} | {d['probe']:.1f} | {d['probe_cv']:.3f} | {d['majority']:.1f} |")
open(f"{FS}/readouts_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(json.dumps({k: out[k] for k in ["reader_valid", "control", "retest", "calls_used", "ledger_status"]}, indent=1))
