"""FS-E1 reader output parser: take the first JSON object in the output (json.JSONDecoder.raw_decode from the first '{'),
accept ids as "Q5", "5" or 5, and answers yes/no in any case. Missing or other answers stay None (counted wrong)."""
import json, re


def parse(out, qids):
    dec = json.JSONDecoder(); obj = None
    for m in re.finditer(r"\{", out):
        try:
            obj, _ = dec.raw_decode(out[m.start():]); break
        except Exception:
            continue
    ans = {}
    try:
        for a in obj["answers"]:
            k = str(a["id"]).strip().upper(); k = k if k.startswith("Q") else f"Q{k}"
            ans[k] = str(a["answer"]).strip().lower()
    except Exception:
        pass
    ans = {q: (ans.get(q) if ans.get(q) in ("yes", "no") else None) for q in qids}
    return ans, all(v is not None for v in ans.values())
