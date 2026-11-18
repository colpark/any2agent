"""FB-E1: re-run the fixed audit on a saved agent call; if clean and valid, keep the output (no new Astra call)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fb3_agent import audit, SCHEMA, FS, FB
import jsonschema
call, raw = sys.argv[1], sys.argv[2]
cmds = [c["command"] for c in json.load(open(f"{FS}/fb/{raw}/commands.json"))]
flags = audit(f"{FB}/ws_{call}", cmds)
obj = json.load(open(f"{FS}/fb/{raw}/last_message.json")); errs = [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(obj)]
for t in obj["answers"]:
    if [e["id"] for e in t["entries"]] != [f"Q{k}" for k in range(1, 18)]: errs.append("ids")
rec = {"call": call, "raw_dir": raw, "reaudit": "FB-E1", "n_commands": len(cmds), "audit_flags": flags, "errors": errs,
       "status": "ok" if not flags and not errs else "voided_audit" if flags else "invalid"}
open(f"{FS}/fb/reaudit.jsonl", "a").write(json.dumps(rec) + "\n")
if rec["status"] == "ok": json.dump(obj, open(f"{FS}/fb/agent_{call}.json", "w"), indent=1)
print(json.dumps(rec, indent=1))
