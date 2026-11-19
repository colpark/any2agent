"""AH-2 Astra calls. Usage: ah2.py code | apply | free
code:  1 call (retry once on any failure, cap 4 in total) in a fresh copy of ws_code, read-only sandbox, 45 min, stage 1 audit + static check.
apply: run label(desc, x) in a subprocess on the bank (500) and Test (997) -> ~/langres_state/ah/code_{bank,test}.npy (1/0/-1 abstain).
free:  1 call (retry once) in a fresh copy of ws_free, workspace-write, network off, 60 min; validates scores.npy."""
import json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jsonschema
from ah_common import *
from th_common import check_code
from th1_code import SCALING
CODE_PROMPT = ("Write Python code that labels Sentinel-2 L2A patches with the 15 CORINE Land Cover classes in classes.json, using physical reasoning about band "
               "values only. No labels exist. bank_descriptors.npy holds ten band statistics for 500 unlabeled patches, with names and formulas in "
               "descriptors.json. sample_bands.npy holds 50 of those patches as 12 × 120 × 120 arrays, {scaling}, band order B01, B02, B03, B04, B05, B06, "
               "B07, B08, B8A, B09, B11, B12, all at 10 m, so each patch covers 1.2 km. Use them to check that your code runs and to set thresholds from the "
               "distribution. Define label(desc, x), taking a dict of the ten descriptors and one 12 × 120 × 120 array, and returning a dict from class "
               "number to \"yes\", \"no\" or \"abstain\". Commit to yes or no whenever the physics gives real evidence, and abstain otherwise. Use only numpy "
               "and scipy, with no file, network or system access. For each class, also state the evidence your rule uses in at most 15 words.").replace("{scaling}", SCALING)
CODE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["code", "evidence"], "properties": {"code": {"type": "string"},
               "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "text"],
                                                        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}}}}}}
FREE_PROMPT = ("You will predict land cover for 997 Sentinel-2 test patches, each 1.2 km on a side, without any labels. The working folder holds, for 500 bank "
               "patches and the 997 test patches: Prithvi-EO-2.0 foundation model features (4,096 standardized numbers per patch), ten physical band "
               "statistics per patch (named in descriptors.json), a structured description of each patch written by a vision language model from true and "
               "false color images, and that model's yes or no answers with confidence for each class. For the bank only, bank_astra.jsonl holds answers "
               "from a stronger vision language model. classes.json lists the 15 CORINE Land Cover classes with definitions. No true labels exist anywhere, "
               "and you must not look for any outside this folder. Use any method you like, such as combining the answer sources, training models on the "
               "features with pseudo labels, or physical rules. Python with numpy, scipy and scikit-learn is available. Write scores.npy, shape 997 × 15, "
               "each entry your probability that the class is present, with rows in the order of test_ids.json and columns in the order of classes.json. "
               "Also write method.md describing your method in at most 300 words.")
FREE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["done", "summary"], "properties": {"done": {"type": "boolean"}, "summary": {"type": "string"}}}
RUNNER = """import json, sys, numpy as np
src = open(sys.argv[1]).read(); X = np.load(sys.argv[2], mmap_mode="r"); D = np.load(sys.argv[3]); names = json.load(open(sys.argv[4])); g = {"__name__": "rules"}
exec(compile(src, "rule_code", "exec"), g)
out = []
for i in range(len(X)):
    try:
        r = g["label"]({n: float(D[i, j]) for j, n in enumerate(names)}, np.asarray(X[i], dtype=np.float32)); out.append({str(int(k)): str(v) for k, v in r.items()})
    except Exception as e:
        out.append({"error": repr(e)[:200]})
json.dump(out, open(sys.argv[5], "w"))
"""


def call(kind):
    for attempt in (1, 2):
        if used() >= CAP: sys.exit(f"cap reached {used()}/{CAP}")
        n = used() + 1; name = f"a{n}_{kind}_try{attempt}"; ws = f"{ST}/ws_{kind}_run{n}"; shutil.copytree(f"{ST}/ws_{kind}", ws)
        schema, prompt = (CODE_SCHEMA, CODE_PROMPT) if kind == "code" else (FREE_SCHEMA, FREE_PROMPT)
        json.dump(schema, open(f"{ST}/schema_{kind}.json", "w"), indent=1)
        rec, out = codex(name, prompt, ws, f"{ST}/schema_{kind}.json", sandbox="read-only" if kind == "code" else "workspace-write",
                         timeout=2700 if kind == "code" else 3600)
        errs, notes = [], {}
        try:
            obj = json.load(open(f"{out}/last_message.json")); errs += [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(obj)]
            if kind == "code":
                if sorted(e["id"] for e in obj["evidence"]) != CL: errs.append("evidence ids != the 15 classes")
                errs += check_code(obj["code"]); notes["evidence_over_15_words"] = [e["id"] for e in obj["evidence"] if len(e["text"].split()) > 15]
            else:
                if len(obj["summary"].split()) > 100: errs.append("summary over 100 words")
                S = np.load(f"{ws}/scores.npy"); shutil.copy(f"{ws}/scores.npy", f"{out}/scores.npy")
                if S.shape != (997, 15) or not np.isfinite(S).all() or S.min() < 0 or S.max() > 1: errs.append(f"scores.npy invalid: shape {S.shape}")
                if os.path.exists(f"{ws}/method.md"):
                    shutil.copy(f"{ws}/method.md", f"{out}/method.md"); notes["method_words"] = len(open(f"{ws}/method.md").read().split())
                else: errs.append("no method.md")
                notes["workspace_files_after"] = sorted(os.listdir(ws))
        except Exception as e:
            errs.append(f"parse/validate: {e}")
        rec.update({"attempt": attempt, "errors": errs, "notes": notes, "workspace": ws,
                    "status": "usage_limit" if rec["usage_limit"] else "voided_audit" if rec["audit_flags"] else "invalid" if errs else "ok"})
        open(LEDGER, "a").write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("name", "status", "latency_s", "n_commands", "audit_flags", "errors", "notes")}, indent=1), flush=True)
        if rec["status"] == "ok":
            if kind == "code": json.dump(obj, open(f"{AH}/code_rules.json", "w"), indent=1); open(f"{AH}/code_rules.py", "w").write(obj["code"])
            else: shutil.copy(f"{out}/scores.npy", f"{ST}/free_scores.npy")
            return
        if rec["status"] == "usage_limit": sys.exit("usage limit: stop and report")


def apply():
    S = sets(); obj = json.load(open(f"{AH}/code_rules.json")); assert not check_code(obj["code"]); res = {}
    names = json.load(open(f"{ST}/descriptors.json"))["names"]
    for tag, split, idx in (("bank", "rlpool", S["bank_rl"]), ("test", "test", S["test"])):
        X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r"); arr = f"{ST}/code_input_{tag}.npy"; np.save(arr, np.stack([np.asarray(X[i], np.float32) for i in idx]))
        with tempfile.TemporaryDirectory(dir=ST) as td:
            open(f"{td}/run.py", "w").write(RUNNER); open(f"{td}/names.json", "w").write(json.dumps(names)); t0 = time.time()
            p = subprocess.run([PY, "-I", f"{td}/run.py", f"{AH}/code_rules.py", arr, f"{ST}/desc_{tag}.npy", f"{td}/names.json", f"{ST}/code_labels_{tag}.json"],
                               capture_output=True, text=True, timeout=3600, cwd=td)
        os.remove(arr); L = json.load(open(f"{ST}/code_labels_{tag}.json"))
        M = np.array([[{"yes": 1, "no": 0}.get(r.get(str(c)), -1) for c in CL] for r in L], np.int8); np.save(f"{ST}/code_{tag}.npy", M)
        res[tag] = {"returncode": p.returncode, "sec": round(time.time() - t0), "errors": sum("error" in r for r in L), "stderr": p.stderr[-500:],
                    "coverage": {c: float((M[:, j] >= 0).mean()) for j, c in enumerate(CL)}, "yes_rate": {c: float((M[:, j] == 1).mean()) for j, c in enumerate(CL)}}
    json.dump(res, open(f"{AH}/code_apply.json", "w"), indent=1); print(json.dumps(res, indent=1))


PY = f"{H}/langres_state/venv/bin/python"
if __name__ == "__main__":
    {"code": lambda: call("code"), "free": lambda: call("free"), "apply": apply}[sys.argv[1]]()
