"""FB-3: Astra agent calls A1 (captions + features) and A0 (captions only) through codex exec, plus the command audit.
Usage: fb3_agent.py --call A1|A0 [--prepare_only]
Cap: 3 Astra calls in total for FB (ledger frontier_share/fb/calls.jsonl), retries included, at most one retry overall.
The workspace holds only the files named in DESIGN_FB_v1; it never holds patch ids, labels, band values or keys."""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
import numpy as np
import jsonschema

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; FB = f"{H}/langres_state/fb"; PR = f"{H}/langres_data/prithvi"
PY = f"{H}/langres_state/venv/bin/python"; CAP = 3; LEDGER = f"{FS}/fb/calls.jsonl"
SANDBOX = ["-s", "read-only"]  # frozen v1 setting; see CHANGE_REQUEST_FB-X1

A1 = ("You will answer questions about 10 target Sentinel-2 satellite patches, each 1.2 km on a side. The working folder holds: "
      "bank_captions.jsonl, short descriptions of 1,000 other patches written by another vision language model from true color and false color images. "
      "bank_features.npy, a 1,000 × 1024 array of Prithvi-EO-2.0 foundation model features for those patches, computed from six Sentinel-2 bands "
      "including shortwave infrared (B11, B12), which no description saw. Rows match the order of bank_captions.jsonl. Features are standardized per dimension. "
      "target_captions.jsonl, detailed descriptions of the 10 targets written by you from the same kind of renderings. "
      "target_features.npy, the targets' features, with rows in the same order. questions.json, 17 yes or no questions. "
      "No labels, band values or answer keys exist in this folder, and you must not look for any outside it. Use the descriptions and any analysis "
      "of the features you find useful, such as similarity, clustering or relating feature directions to what the bank descriptions say. "
      f"Python with numpy and scikit-learn is available at {PY}. Answer every question for every target with yes or no and a confidence of low, "
      "medium or high, and give your best guess when uncertain. Also describe your method in at most 200 words.")
A0 = ("You will answer questions about 10 target Sentinel-2 satellite patches, each 1.2 km on a side. The working folder holds: "
      "bank_captions.jsonl, short descriptions of 1,000 other patches written by another vision language model from true color and false color images. "
      "target_captions.jsonl, detailed descriptions of the 10 targets written by you from the same kind of renderings. "
      "questions.json, 17 yes or no questions. "
      "No labels, band values or answer keys exist in this folder, and you must not look for any outside it. Use the descriptions and any analysis "
      "you find useful. "
      f"Python with numpy and scikit-learn is available at {PY}. Answer every question for every target with yes or no and a confidence of low, "
      "medium or high, and give your best guess when uncertain. Also describe your method in at most 200 words.")
PROMPTS = {"A1": A1, "A0": A0}

ENTRY = {"type": "object", "properties": {"id": {"type": "string"}, "answer": {"type": "string", "enum": ["yes", "no"]},
                                          "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
         "required": ["id", "answer", "confidence"], "additionalProperties": False}
TARGET = {"type": "object", "properties": {"target_id": {"type": "string"}, "entries": {"type": "array", "items": ENTRY}},
          "required": ["target_id", "entries"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "items": TARGET}, "method": {"type": "string"}},
          "required": ["answers", "method"], "additionalProperties": False}


def prepare(call):
    ws = f"{FB}/ws_{call}"
    if os.path.exists(ws): shutil.rmtree(ws)
    os.makedirs(ws)
    bank = json.load(open(f"{FB}/bank.json")); caps = [json.loads(l) for l in open(f"{FB}/bank_captions.jsonl")]
    sel = json.load(open(f"{H}/langres_state/fs/selection.json")); U, TI = sel["uid_order"], sel["test100_index"]
    with open(f"{ws}/bank_captions.jsonl", "w") as f:
        for c in caps: f.write(json.dumps({"bank_id": c["bank_id"], "caption": c["caption"]}) + "\n")
    with open(f"{ws}/target_captions.jsonl", "w") as f:
        for u in U: f.write(json.dumps({"target_id": u, "caption": json.load(open(f"{FS}/raw/caption_main_{u}.json"))}, ensure_ascii=False) + "\n")
    json.dump([{"id": q["id"], "text": q["text"]} for q in json.load(open(f"{FS}/fb/questions17.json"))], open(f"{ws}/questions.json", "w"), indent=1)
    if call == "A1":
        np.save(f"{ws}/bank_features.npy", np.load(f"{PR}/rlpool_prithvi.npy")[bank["rlpool_index"]].astype(np.float32))
        np.save(f"{ws}/target_features.npy", np.load(f"{PR}/test100_prithvi.npy")[TI].astype(np.float32))
    json.dump(SCHEMA, open(f"{FB}/schema_agent.json", "w"), indent=1)
    return ws, {f: hashlib.sha256(open(f"{ws}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(ws))}


NET = re.compile(r"\b(curl|wget|pip|ssh|scp|git|nc|http[s]?://|socket|requests|urllib|ftp)\b", re.I)


def audit(ws, commands):
    """FB-E1: a path is a token that starts at a word boundary (start, space, quote, =, () with "/" plus a real top-level
    directory, or with ~, $HOME or ../. Arithmetic such as x/np.sqrt(y) is not a path."""
    allowed = (ws, f"{H}/langres_state/venv", "/usr/lib/python3", "/usr/bin/", "/bin/", "/dev/null")
    roots = "|".join(sorted(d for d in os.listdir("/") if d))
    pat = re.compile(rf"(?:(?<=^)|(?<=[\s'\"=(]))(/(?:{roots})(?:/[^\s'\"<>|;,)]*)?|~[^\s'\"<>|;,)]*|\$HOME[^\s'\"<>|;,)]*|\.\./[^\s'\"<>|;,)]*)")
    flags = []
    for c in commands:
        for tok in pat.findall(c):
            if not tok.startswith(allowed):
                flags.append(("outside_path", tok, c[:200]))
        if NET.search(c):
            flags.append(("network", NET.search(c).group(0), c[:200]))
    return flags


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--call", required=True, choices=["A1", "A0"]); ap.add_argument("--prepare_only", action="store_true")
    a = ap.parse_args()
    ws, hashes = prepare(a.call); print("workspace", ws, json.dumps(hashes, indent=1))
    if a.prepare_only: sys.exit(0)
    if used() >= CAP: sys.exit(f"cap reached {used()}/{CAP}")
    out = f"{FS}/fb/raw_{a.call}_try{used() + 1}"; os.makedirs(out)
    cmd = ["timeout", "2700", "codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
           "-m", "gpt-6-astra", *SANDBOX, "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="high"', "-c", 'web_search="disabled"',
           "--output-schema", f"{FB}/schema_agent.json", "--json", "-o", f"{out}/last_message.json", "-C", ws, "--color", "never"]
    t0 = time.time(); p = subprocess.run(cmd, input=PROMPTS[a.call], capture_output=True, text=True, cwd=ws); lat = round(time.time() - t0, 1)
    open(f"{out}/events.jsonl", "w").write(p.stdout); open(f"{out}/stderr.txt", "w").write(p.stderr)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    cmds = [e["item"] for e in ev if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "command_execution"]
    other_tools = sorted({e["item"]["type"] for e in ev if "item" in e} - {"agent_message", "reasoning", "command_execution", "todo_list"})
    commands = [c.get("command", "") for c in cmds]
    json.dump(cmds, open(f"{out}/commands.json", "w"), indent=1)
    flags = audit(ws, commands) + [("tool", t, "") for t in other_tools if t in ("web_search", "mcp_tool_call", "file_change")]
    txt = open(f"{out}/last_message.json").read() if os.path.exists(f"{out}/last_message.json") else ""
    errs = []
    try:
        obj = json.loads(txt); errs = [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(obj)]
        tids = [t["target_id"] for t in obj["answers"]]; U = [json.loads(l)["target_id"] for l in open(f"{ws}/target_captions.jsonl")]
        if sorted(tids) != sorted(U): errs.append("target ids do not match")
        for t in obj["answers"]:
            if [e["id"] for e in t["entries"]] != [f"Q{k}" for k in range(1, 18)]: errs.append(f"{t['target_id'][:8]} ids not Q1..Q17 in order")
        if len(obj["method"].split()) > 200: errs.append("method over 200 words")
    except Exception as e:
        obj, errs = None, [f"parse: {e}"]
    limit = p.returncode != 0 and any(w in (p.stdout + p.stderr).lower() for w in ("usage limit", "usage_limit", "rate limit", "quota"))
    status = "usage_limit" if limit else "timeout" if p.returncode == 124 else "voided_audit" if flags else "invalid" if errs else "ok"
    rec = {"call": a.call, "try": used() + 1, "status": status, "returncode": p.returncode, "latency_s": lat, "n_commands": len(commands),
           "failed_commands": sum(1 for c in cmds if c.get("exit_code") not in (0, None)), "audit_flags": flags, "errors": errs[:5],
           "prompt_sha256": hashlib.sha256(PROMPTS[a.call].encode()).hexdigest(), "workspace_sha256": hashes, "cmd": " ".join(cmd),
           "usage": [e.get("usage") for e in ev if e.get("type") == "turn.completed"], "raw_dir": os.path.relpath(out, FS),
           "t_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    open(LEDGER, "a").write(json.dumps(rec) + "\n")
    if status == "ok": json.dump(obj, open(f"{FS}/fb/agent_{a.call}.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("call", "try", "status", "latency_s", "n_commands", "failed_commands", "audit_flags", "errors")}, indent=1))
