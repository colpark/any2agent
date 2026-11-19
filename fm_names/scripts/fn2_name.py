"""FN-2: 20 Astra naming calls (stage-1 image-QA settings via th_common.codex_call, agent=False: any tool/command voids).
Cap 24 incl. retries (ledger fm_names/calls.jsonl), at most 3 concurrent, stop on usage limit. Writes fm_names/names.json."""
import json, os, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import jsonschema
from fn_common import *

PROMPT = ("You see twelve Sentinel-2 L2A patches, each 1.2 km on a side with north up. Each image shows one patch, with true color (B04, B03, B02) "
          "on the left and false color (B08, B04, B03), where near infrared appears red, on the right. A foundation model that also sees shortwave "
          "infrared places patches A1 to A6 (images 1 to 6) at one end of one of its internal feature directions and patches B1 to B6 (images 7 to 12) "
          "at the other end. Work out what separates group A from group B. Answer from the images only. Do not run commands or use tools. Give a short "
          "name for each end, a definition that lets another vision language model judge from the same kind of image which of two new patches lies "
          "closer to end A, and up to four visible cues to check, most telling first. Rate how visible the difference is: low, medium or high.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name_a", "name_b", "definition", "cues", "visible"], "properties": {
    "name_a": {"type": "string"}, "name_b": {"type": "string"}, "definition": {"type": "string"},
    "cues": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}, "visible": {"type": "string", "enum": ["low", "medium", "high"]}}}
F1 = json.load(open(f"{FN}/fn1.json")); LOCK = threading.Lock(); STOP = threading.Event(); PENDING = set()


def one(d, attempt=1):
    with LOCK:
        n = used() + len(PENDING)
        if n >= CAP or STOP.is_set(): return None
        cid = f"n{n + 1:02d}"; PENDING.add(cid)
    ws = tempfile.mkdtemp(dir=ST, prefix=f"ws_{cid}_"); imgs = F1["exemplar_images"][str(d)]["A"] + F1["exemplar_images"][str(d)]["B"]
    rec, out = codex_call(f"{cid}_pc{d:02d}", PROMPT, ws, f"{ST}/schema_name.json", images=imgs, agent=False)
    try:
        obj = json.load(open(f"{out}/last_message.json")); errs = [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(obj)]
    except Exception as e:
        obj, errs = None, [f"parse: {e}"]
    rec.update({"call_id": cid, "pc": d, "attempt": attempt, "errors": errs[:5], "workdir_empty_after": os.listdir(ws) == []})
    rec["status"] = "usage_limit" if rec["usage_limit"] else "voided_tool" if rec["audit_flags"] else "invalid" if errs else "ok"
    if rec["usage_limit"]: STOP.set()
    if obj and not errs: rec["word_counts"] = {"name_a": len(obj["name_a"].split()), "name_b": len(obj["name_b"].split()), "definition": len(obj["definition"].split())}
    with LOCK:
        open(LEDGER, "a").write(json.dumps(rec) + "\n"); PENDING.discard(cid)
    print(json.dumps({k: rec.get(k) for k in ("call_id", "pc", "attempt", "status", "latency_s", "errors", "word_counts")}), flush=True)
    if rec["status"] in ("invalid", "voided_tool") and attempt == 1 and not STOP.is_set(): return one(d, 2)
    return (d, obj) if rec["status"] == "ok" else None


if __name__ == "__main__":
    json.dump(SCHEMA, open(f"{ST}/schema_name.json", "w"), indent=1)
    p = f"{FN}/names.json"; names = json.load(open(p)) if os.path.exists(p) else {}
    todo = [d for d in range(20) if str(d) not in names]
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(one, todo):
            if r: names[str(r[0])] = {"pc": r[0], "pc_rank": r[0] + 1, "a_end": F1["ends"][str(r[0])], **r[1]}
    json.dump(dict(sorted(names.items(), key=lambda kv: int(kv[0]))), open(p, "w"), indent=1)
    print(f"named {len(names)}/20; calls used {used()}/{CAP}; usage_limit={STOP.is_set()}")
