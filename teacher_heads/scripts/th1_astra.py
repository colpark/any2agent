"""TH-1 Astra VLM teacher: 5 patches per call, pilot image-QA settings, any tool/command voids the call.
Usage: th1_astra.py check | bank | test200.  check = 2 calls on Pilot10 + agreement with the pilot's single-patch Q5..Q10.
Patch order shuffled with seed 0 per set; patch k gets images 2k-1 (true color), 2k (false color). Ledger: teacher_heads/calls.jsonl."""
import json, os, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jsonschema
from th_common import *

PROMPT = ("You see five Sentinel-2 L2A patches, P1 to P5. Each is 1.2 km on a side with north up and has two images in order: first true color "
          "(B04, B03, B02), then false color (B08, B04, B03), where near infrared appears red. Images 1 and 2 show P1, images 3 and 4 show P2, and so on. "
          "Answer from the images only. Do not run commands or use tools. For each patch and each land cover class below, answer whether any part of "
          "the patch belongs to it, following the CORINE Land Cover nomenclature, with yes or no and a confidence of low, medium or high. "
          "Judge each patch on its own. ")
PIDS = ["P1", "P2", "P3", "P4", "P5"]
ANS = {"type": "object", "additionalProperties": False, "required": ["id", "answer", "confidence"], "properties": {
    "id": {"type": "integer"}, "answer": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["patches"], "properties": {"patches": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["pid", "answers"], "properties": {
        "pid": {"type": "string", "enum": PIDS}, "answers": {"type": "array", "items": ANS}}}}}}
LOCK = threading.Lock(); STOP = threading.Event(); PENDING = set()


def validate(obj):
    errs = [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(obj)]
    if not errs:
        if sorted(p["pid"] for p in obj["patches"]) != PIDS: errs.append("pids != P1..P5")
        for p in obj["patches"]:
            if sorted(a["id"] for a in p["answers"]) != CID: errs.append(f"{p['pid']} ids != 1..19")
    return errs


def one(batch, imgs, tag, k, attempt=1):
    with LOCK:
        n = used() + len(PENDING)
        if n >= CAP or STOP.is_set(): return None
        cid = f"c{n + 1:03d}"; PENDING.add(cid)
    ws = tempfile.mkdtemp(dir=ST, prefix=f"ws_{cid}_")
    rec, out = codex_call(f"{cid}_{tag}_b{k:03d}", PROMPT + class_block(), ws, f"{ST}/schema_astra.json",
                          images=[p for i in batch for p in imgs[i]], agent=False)
    try:
        obj = json.load(open(f"{out}/last_message.json")); errs = validate(obj)
    except Exception as e:
        obj, errs = None, [f"parse: {e}"]
    rec.update({"call_id": cid, "set": tag, "batch": k, "attempt": attempt, "patch_index": [int(i) for i in batch], "errors": errs[:5],
                "workdir_empty_after": os.listdir(ws) == []})
    rec["status"] = "usage_limit" if rec["usage_limit"] else "voided_tool" if rec["audit_flags"] else "invalid" if errs else "ok"
    if rec["usage_limit"]: STOP.set()
    with LOCK:
        open(LEDGER, "a").write(json.dumps(rec) + "\n"); PENDING.discard(cid)
    print(json.dumps({x: rec[x] for x in ("call_id", "set", "batch", "attempt", "status", "latency_s", "errors")}), flush=True)
    if rec["status"] in ("invalid", "voided_tool") and attempt == 1 and not STOP.is_set():
        return one(batch, imgs, tag, k, 2)
    if rec["status"] != "ok": return None
    return {int(i): {str(a["id"]): a["answer"] for a in p["answers"]} for i, p in zip(batch, sorted(obj["patches"], key=lambda p: p["pid"]))}


def run(tag, split, idx):
    json.dump(SCHEMA, open(f"{ST}/schema_astra.json", "w"), indent=1)
    done_p = f"{ST}/astra_{tag}.json"; done = {int(k): v for k, v in json.load(open(done_p)).items()} if os.path.exists(done_p) else {}
    order = np.random.default_rng(0).permutation(np.array(idx)); batches = [order[k:k + 5] for k in range(0, len(order), 5)]
    imgs = render(split, [i for i in order if int(i) not in done])
    todo = [(k, b) for k, b in enumerate(batches) if any(int(i) not in done for i in b)]
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(lambda kb: one(kb[1], imgs, tag, kb[0]), todo):
            if r: done.update(r); json.dump({str(i): a for i, a in done.items()}, open(done_p, "w"))
    print(f"{tag}: {len(done)}/{len(idx)} patches answered; calls used {used()}/{CAP}; usage_limit={STOP.is_set()}")
    return done


if __name__ == "__main__":
    s = sets(); m = sys.argv[1]
    if m == "check":
        A = run("pilot10", "test", s["pilot10"])
        qmap = {5: 3, 6: 11, 7: 10, 8: 14, 9: 9, 10: 7}; agree = []
        for u, i in zip(s["pilot10_uid"], s["pilot10"]):
            q = {a["id"]: a["answer"] for a in json.load(open(f"{REPO}/frontier_share/raw/qa_main_{u}.json"))["answers"]}
            if i in A: agree += [A[i][str(c)] == q[f"Q{n}"] for n, c in qmap.items()]
        res = {"n_compared": len(agree), "agreement": float(np.mean(agree)) if agree else None}
        res["pass"] = bool(agree) and len(agree) == 60 and res["agreement"] >= 0.8
        json.dump(res, open(f"{TH}/batching_check.json", "w"), indent=1); print("BATCHING_CHECK", res)
    elif m == "bank":
        run("bank", "rlpool", s["bank_rl"])
    elif m == "test200":
        run("test200", "test", s["test200"])
