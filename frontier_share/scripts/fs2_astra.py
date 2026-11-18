"""FS-2: GPT-6 Astra calls through headless `codex exec` (ChatGPT subscription login, no API key).
Usage: fs2_astra.py --kind caption|qa --role main|retest --uids U1 U2 ...
Every invocation is appended to frontier_share/calls.jsonl and counts against the hard cap of 35 (retries included).
Each call runs in a fresh empty directory; the JSON event stream, the last message and stderr are all saved.
A call that shows any tool, command, web search or file change is discarded and counted as failed.
An invalid output (schema or word budget) is retried once, within the cap. Any usage-limit message stops everything.
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
import jsonschema

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; TMP = f"{H}/langres_state/fs/tmp"
CAP, CONC, MODEL, EFFORT = 35, 3, "gpt-6-astra", "high"
LEDGER = f"{FS}/calls.jsonl"; LOCK = threading.Lock(); STOP = threading.Event()

CAPTION_PROMPT = """You are an expert in Earth observation and remote sensing. You see two renderings of the same Sentinel-2 L2A patch: 120 × 120 pixels at 10 m, so 1.2 km on each side, upsampled for display. North is up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears as red, so healthy vegetation appears bright red.

Answer from the two attached images only. Do not run commands or use tools. Do not name a location unless the image itself shows it, and use no outside data. Give every inference a confidence of low, medium or high.

Work through these levels in order and fill the JSON schema.

1. Global scene. The dominant land covers with estimated area percentages summing to 100, the landscape type, cues of season or phenology, and overall heterogeneity.
2. Regional layout. Split the patch into a 3 × 3 grid: rows top, middle, bottom, and columns left, center, right. For each cell give the dominant cover, the texture and notable features. Then describe the boundaries and gradients across cells.
3. Local features. List each discrete feature, such as fields, roads, rivers, lakes, buildings, clearings, clouds and shadows, with its type, grid cell, approximate size in meters, shape, orientation and count.
4. Radiometric and spectral cues. Describe brightness, color and contrast in true color, and the strength and pattern of red in false color. State what they imply about vegetation vigor, moisture, water turbidity, bare soil and built surfaces. State what these two renderings cannot reveal.
5. Scientific interpretation. Land use and management (crop types, rotation, forestry, settlement), ecological state, active processes (irrigation, harvest, erosion, flooding, clearing), and the likely biome and climate setting. Give each claim its confidence and its evidence as a level and grid cell.
6. Uncertainties. Features you cannot resolve at 10 m, ambiguous classes and possible artifacts.

Then write nested summaries for a scientist of at most 25, 50, 100, 200 and 400 words. Each summary stands alone in plain prose and carries the most informative content its budget allows. Longer summaries add finer detail and never contradict shorter ones. Return only JSON that matches the schema."""

QA_HEAD = """You see two renderings of the same Sentinel-2 L2A patch: 120 × 120 pixels at 10 m, so 1.2 km on each side, upsampled for display. North is up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears as red. Answer from the two attached images only. Do not run commands or use tools. Answer every question below with yes or no and a confidence of low, medium or high. When the images do not settle a question, give your best guess."""
QUESTIONS = json.load(open(f"{FS}/questions.json"))
QA_PROMPT = QA_HEAD + "\n\n" + "\n".join(f"{q['id']}. {q['text']}" for q in QUESTIONS)
PROMPTS = {"caption": CAPTION_PROMPT, "qa": QA_PROMPT}
SCHEMAS = {"caption": f"{FS}/schemas/caption.json", "qa": f"{FS}/schemas/qa.json"}
BUDGET = {"w25": 25, "w50": 50, "w100": 100, "w200": 200, "w400": 400}
TOOL_ITEMS = {"command_execution", "mcp_tool_call", "web_search", "file_change", "collab_tool_call", "image_generation"}
LIMIT_WORDS = ("usage_limit", "usage limit", "rate_limit", "rate limit", "quota", "too many requests")


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


def reserve():
    with LOCK:
        n = used() + len(PENDING)
        if n >= CAP or STOP.is_set():
            return None
        cid = f"c{n + 1:02d}"; PENDING.add(cid); return cid


PENDING = set()


def validate(kind, txt):
    try:
        obj = json.loads(txt)
    except Exception as e:
        return None, [f"json: {e}"]
    errs = [e.message for e in jsonschema.Draft202012Validator(json.load(open(SCHEMAS[kind]))).iter_errors(obj)]
    if kind == "caption" and not errs:
        for k, b in BUDGET.items():
            n = len(obj["summaries"][k].split())
            if n > b: errs.append(f"{k} has {n} words > {b}")
    if kind == "qa" and not errs:
        ids = [a["id"] for a in obj["answers"]]
        if ids != [q["id"] for q in QUESTIONS]: errs.append(f"answer ids {ids} do not match Q1..Q16 in order")
    return obj, errs


def call(kind, role, u, attempt):
    cid = reserve()
    if cid is None:
        return {"uid": u, "kind": kind, "role": role, "status": "not_run_cap_or_stop"}
    wd = f"{TMP}/{cid}_{uuid.uuid4().hex[:8]}"; os.makedirs(wd)
    out_dir = f"{FS}/raw/{cid}_{kind}_{role}_{u[:8]}"; os.makedirs(out_dir)
    last = f"{out_dir}/last_message.json"
    cmd = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
           "-m", MODEL, "-s", "read-only", "-c", 'approval_policy="never"', "-c", f'model_reasoning_effort="{EFFORT}"',
           "-c", 'web_search="disabled"', "--output-schema", SCHEMAS[kind], "--json", "-o", last, "-C", wd, "--color", "never",
           "-i", f"{FS}/renders/{u}_tc.png", f"{FS}/renders/{u}_fc.png"]
    prompt = PROMPTS[kind]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3600, cwd=wd)
    lat = round(time.time() - t0, 1)
    open(f"{out_dir}/events.jsonl", "w").write(p.stdout); open(f"{out_dir}/stderr.txt", "w").write(p.stderr)
    ev = []
    for line in p.stdout.splitlines():
        try: ev.append(json.loads(line))
        except Exception: pass
    item_types = sorted({e.get("item", {}).get("type", "") for e in ev if "item" in e} - {""})
    tools = [t for t in item_types if t in TOOL_ITEMS]
    usage = [e.get("usage") for e in ev if e.get("type") == "turn.completed"]
    blob = (p.stdout + p.stderr).lower()
    limit = any(w in blob for w in LIMIT_WORDS) and p.returncode != 0
    txt = open(last).read() if os.path.exists(last) else ""
    obj, errs = validate(kind, txt) if txt else (None, ["no last message"])
    status = "ok"
    if p.returncode != 0: status = "exit_error"
    if limit: status = "usage_limit"; STOP.set()
    if tools: status = "tool_use_discarded"
    elif status == "ok" and errs: status = "invalid"
    rec = {"call_id": cid, "kind": kind, "role": role, "uid": u, "attempt": attempt, "model": MODEL, "effort": EFFORT,
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "schema_sha256": hashlib.sha256(open(SCHEMAS[kind], "rb").read()).hexdigest(),
           "cmd": " ".join(cmd), "returncode": p.returncode, "latency_s": lat, "status": status, "errors": errs[:5],
           "item_types": item_types, "usage": usage, "workdir_empty_after": os.listdir(wd) == [], "raw_dir": os.path.relpath(out_dir, FS),
           "t_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with LOCK:
        open(LEDGER, "a").write(json.dumps(rec) + "\n"); PENDING.discard(cid)
    if status == "ok":
        json.dump(obj, open(f"{FS}/raw/{kind}_{role}_{u}.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("call_id", "kind", "role", "uid", "attempt", "status", "latency_s", "errors", "item_types")}), flush=True)
    return rec


def job(kind, role, u):
    r = call(kind, role, u, 1)
    if r.get("status") in ("invalid", "tool_use_discarded", "exit_error") and not STOP.is_set():
        r = call(kind, role, u, 2)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["caption", "qa"])
    ap.add_argument("--role", required=True, choices=["main", "retest"]); ap.add_argument("--uids", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True); os.makedirs(f"{FS}/raw", exist_ok=True)
    todo = [u for u in a.uids if not os.path.exists(f"{FS}/raw/{a.kind}_{a.role}_{u}.json")]
    print(f"calls used {used()}/{CAP}; jobs {len(todo)}", flush=True)
    with ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(lambda u: job(a.kind, a.role, u), todo))
    print(f"calls used {used()}/{CAP}; usage_limit={STOP.is_set()}", flush=True)
    sys.exit(3 if STOP.is_set() else 0)
