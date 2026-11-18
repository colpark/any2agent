"""TH-1 code teacher: workspace (50 unlabeled RL-pool patches outside the bank + classes.json), 1 Astra call (cap 2), audit,
static check, then run label(x) in a subprocess on the bank (1,000) and Test (1,000).
Usage: th1_code.py call | apply"""
import json, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jsonschema
from th_common import *

SCALING = ("raw L2A surface reflectance digital numbers as stored in the BigEarthNet-S2 GeoTIFFs, cast to float32 without rescaling "
           "(reflectance × 10,000), with the 20 m and 60 m bands upsampled bilinearly")
PROMPT = ("Write Python code that labels Sentinel-2 L2A patches with the 19 CORINE land cover classes in classes.json from band values alone, "
          "using your knowledge of remote sensing such as spectral indices, texture and simple spatial statistics. No labels exist. "
          f"sample_bands.npy holds 50 unlabeled patches, shape 50 × 12 × 120 × 120, float32, {SCALING}, band order B01, B02, B03, B04, B05, B06, "
          "B07, B08, B8A, B09, B11, B12, all resampled to 10 m, so each patch covers 1.2 km. Use it only to check that your code runs and gives "
          "plausible class rates. Define label(x) taking one 12 × 120 × 120 array and returning a dict from class number to \"yes\", \"no\" or "
          "\"abstain\". Abstain when the bands cannot settle a class. Use only numpy and scipy, with no file, network or system access. Also write a "
          "one-sentence definition of each class for a vision-language model that sees true color and false color images.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["definitions", "code", "notes"], "properties": {
    "definitions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "name", "definition"],
                    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}, "definition": {"type": "string"}}}},
    "code": {"type": "string"}, "notes": {"type": "string"}}}
RUNNER = """import json, sys, numpy as np
src = open(sys.argv[1]).read(); X = np.load(sys.argv[2], mmap_mode="r"); g = {"__name__": "teacher"}
exec(compile(src, "teacher_code", "exec"), g)
out = []
for i in range(len(X)):
    try:
        r = g["label"](np.asarray(X[i], dtype=np.float32)); out.append({str(int(k)): str(v) for k, v in r.items()})
    except Exception as e:
        out.append({"error": repr(e)[:200]})
json.dump(out, open(sys.argv[3], "w"))
"""


def call():
    s = sets(); rl_ids = json.load(open(f"{ARR}/rlpool_ids.json"))
    outside = np.setdiff1d(np.arange(len(rl_ids)), s["bank_rl"])
    samp = np.sort(np.random.default_rng(0).choice(outside, 50, replace=False))
    ws = f"{ST}/ws_code"; os.makedirs(ws, exist_ok=False)
    X = np.load(f"{ARR}/rlpool_ms12.npy", mmap_mode="r")
    np.save(f"{ws}/sample_bands.npy", np.stack([np.asarray(X[i], np.float32) for i in samp]))
    json.dump([{"id": c, "name": n} for c, n in zip(CID, CLASSES)], open(f"{ws}/classes.json", "w"), indent=1)
    json.dump(SCHEMA, open(f"{ST}/schema_code.json", "w"), indent=1)
    json.dump({"rlpool_index": samp.tolist()}, open(f"{ST}/code_sample.json", "w"))
    rec, out = codex_call(f"code_teacher_try{used() + 1}", PROMPT, ws, f"{ST}/schema_code.json")
    errs = []
    try:
        obj = json.load(open(f"{out}/last_message.json"))
        errs += [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(obj)]
        if sorted(d["id"] for d in obj["definitions"]) != CID: errs.append("definition ids != 1..19")
        if len(obj["notes"].split()) > 150: errs.append("notes over 150 words")
        errs += check_code(obj["code"])
    except Exception as e:
        obj = None; errs.append(f"parse: {e}")
    rec.update({"errors": errs, "status": "usage_limit" if rec["usage_limit"] else "voided_audit" if rec["audit_flags"] else "invalid" if errs else "ok"})
    open(LEDGER, "a").write(json.dumps(rec) + "\n")
    if rec["status"] == "ok":
        json.dump(obj, open(f"{TH}/code_teacher.json", "w"), indent=1); open(f"{TH}/code_teacher.py", "w").write(obj["code"])
    print(json.dumps({k: rec[k] for k in ("name", "try", "status", "latency_s", "n_commands", "audit_flags", "errors")}, indent=1))


def apply():
    s = sets(); obj = json.load(open(f"{TH}/code_teacher.json")); assert not check_code(obj["code"])
    res = {}
    for name, src, idx in [("bank", "rlpool", s["bank_rl"]), ("test", "test", list(range(1000)))]:  # bank = the 500-patch TH bank
        X = np.load(f"{ARR}/{src}_ms12.npy", mmap_mode="r"); arr = f"{ST}/code_input_{name}.npy"
        np.save(arr, np.stack([np.asarray(X[i], np.float32) for i in idx]))
        with tempfile.TemporaryDirectory(dir=ST) as td:
            open(f"{td}/run.py", "w").write(RUNNER); t0 = time.time()
            p = subprocess.run([PY, "-I", f"{td}/run.py", f"{TH}/code_teacher.py", arr, f"{ST}/code_labels_{name}.json"],
                               capture_output=True, text=True, timeout=3600, cwd=td)
        os.remove(arr)
        L = json.load(open(f"{ST}/code_labels_{name}.json"))
        res[name] = {"returncode": p.returncode, "sec": round(time.time() - t0), "errors": sum("error" in r for r in L), "stderr": p.stderr[-500:]}
        M = np.array([[{"yes": 1, "no": 0}.get(r.get(str(c)), -1) for c in CID] for r in L], np.int8)  # -1 = abstain/missing
        np.save(f"{ST}/code_{name}.npy", M)
        res[name]["coverage_per_class"] = {c: float((M[:, k] >= 0).mean()) for k, c in enumerate(CID)}
        res[name]["yes_rate_per_class"] = {c: float((M[:, k] == 1).mean()) for k, c in enumerate(CID)}
    json.dump(res, open(f"{TH}/code_apply.json", "w"), indent=1); print(json.dumps({k: {x: v[x] for x in ("returncode", "sec", "errors")} for k, v in res.items()}))


if __name__ == "__main__":
    {"call": call, "apply": apply}[sys.argv[1]]()
