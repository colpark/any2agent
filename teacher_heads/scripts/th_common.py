"""Shared helpers for TH-0..TH-4 (design DESIGN_TH_v1.md, updated prompt): paths, classes, sets, renders, Codex calls with the
FS/FB settings and audit, code check, grading."""
import ast, hashlib, json, os, re, subprocess, sys, time, uuid
import numpy as np, pandas as pd
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; TH = f"{REPO}/teacher_heads"; ST = f"{H}/langres_state/th"
ARR = f"{H}/langres_data/arrays"; PR = f"{H}/langres_data/prithvi"; PY = f"{H}/langres_state/venv/bin/python"
sys.path.insert(0, f"{REPO}/frontier_share/scripts"); sys.path.insert(0, f"{REPO}/scripts")
from fb3_agent import audit as fb_audit  # FB agent audit (FB-E1 version)
from fs2_astra import TOOL_ITEMS, LIMIT_WORDS  # pilot image-QA call conventions
os.makedirs(ST, exist_ok=True)
CLASSES = ["Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures", "Complex cultivation patterns",
           "Land principally occupied by agriculture, with significant areas of natural vegetation", "Agro-forestry areas",
           "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas",
           "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub", "Beaches, dunes, sands", "Inland wetlands",
           "Coastal wetlands", "Inland waters", "Marine waters"]
CID = list(range(1, 20)); LEDGER = f"{TH}/calls.jsonl"; CAP = 160
NET_CMD = re.compile(r"(^|[\s;|&(])(curl|wget|pip|ssh|scp|git|nc)(\s|$)|https?://|\bsocket\b|\brequests\b|\burllib\b")


def audit(ws, commands):
    """TH-E2: FB audit minus two false-positive classes: numpy '~' (only '~' or '~/...' is a home path) and 'ssh' inside a name."""
    return [f for f in fb_audit(ws, commands) if not (f[0] == "outside_path" and f[1].startswith("~") and not (f[1] == "~" or f[1].startswith("~/")))
            and not (f[0] == "network" and not NET_CMD.search(f[2]))]


def strat(ids, lsets, n, seed=0):
    """make_splits.strat: proportional by exact label set, largest remainder, random within stratum."""
    rng = np.random.default_rng(seed); df = pd.DataFrame({"patch_id": ids, "lset": lsets})
    counts = df["lset"].value_counts().sort_index(); quota = counts / counts.sum() * n
    base = np.floor(quota).astype(int); frac = (quota - base) + rng.uniform(0, 1e-9, len(quota))
    base[frac.sort_values(ascending=False).index[: n - base.sum()]] += 1
    return np.concatenate([rng.choice(np.sort(df.loc[df.lset == ls, "patch_id"].to_numpy()), k, replace=False) for ls, k in base.items() if k > 0])


def sets():
    p = f"{ST}/sets.json"
    if os.path.exists(p): return json.load(open(p))
    d = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id"); ls = lambda ids: ["|".join(sorted(d.loc[i, "labels"])) for i in ids]
    fb = json.load(open(f"{H}/langres_state/fb/bank.json")); rid = json.load(open(f"{ARR}/rlpool_ids.json")); tid = json.load(open(f"{ARR}/test_ids.json"))
    bank = strat(fb["patch_ids"], ls(fb["patch_ids"]), 500); rpos = {p: i for i, p in enumerate(rid)}; tpos = {p: i for i, p in enumerate(tid)}
    t300 = [tid[i] for i in json.load(open(f"{TH}/test300.json"))["test_index"]]
    t200 = strat(t300, ls(t300), 200)
    idm = {r["uid"]: r["patch_id"] for r in json.load(open(f"{REPO}/frontier_share/private/id_map.json"))}
    pil = [idm[u] for u in json.load(open(f"{H}/langres_state/fs/selection.json"))["uid_order"]]
    s = {"bank_rl": sorted(rpos[p] for p in bank), "test300": sorted(tpos[p] for p in t300), "test200": sorted(tpos[p] for p in t200),
         "pilot10": [tpos[p] for p in pil], "pilot10_uid": json.load(open(f"{H}/langres_state/fs/selection.json"))["uid_order"]}
    s["bank_ids"] = [rid[i] for i in s["bank_rl"]]; s["test_ids"] = tid
    json.dump(s, open(p, "w")); return s


def true_labels(ids):  # grading and the label-efficiency reference only
    d = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id")
    return np.array([[CLASSES[c - 1] in ls for c in CID] for ls in d.loc[ids, "labels"]], bool)


def features():
    s = sets(); return np.load(f"{PR}/rlpool_prithvi.npy")[s["bank_rl"]], np.load(f"{PR}/test_prithvi.npy")


def class_block():
    defs = {d["id"]: d["definition"] for d in json.load(open(f"{TH}/code_teacher.json"))["definitions"]}
    return "\n".join(f"{c}. {CLASSES[c - 1]}: {defs[c]}" for c in CID)


def render(split, idx):
    """Pilot renderings: frozen RL-pool 2-98% stretch, 8 bit, PIL bicubic 120 -> 512. Random file names (anonymity)."""
    from PIL import Image
    st = json.load(open(f"{ARR}/stretch_rlpool_p2p98.json")); X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r")
    BI = {b: i for i, b in enumerate(["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"])}
    os.makedirs(f"{ST}/renders", exist_ok=True); out = {}
    for i in idx:
        x = np.asarray(X[i]); name = uuid.uuid4().hex; paths = []
        for tag, bands in (("a", ["B04", "B03", "B02"]), ("b", ["B08", "B04", "B03"])):
            im = np.stack([(x[BI[b]] - st[b][0]) / (st[b][1] - st[b][0]) for b in bands], -1)
            p = f"{ST}/renders/{name}{tag}.png"; Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).resize((512, 512), Image.BICUBIC).save(p)
            paths.append(p)
        out[int(i)] = paths
    return out


def codex_call(name, prompt, ws, schema_path, images=(), agent=True, timeout=2700):
    """One Astra call. agent=True: FB agent settings + audit (venv first on PATH, TH-E1). agent=False: pilot image-QA settings,
    any tool or command voids the call. Counts against CAP (checked by the caller under its lock). Returns (record, raw_dir)."""
    out = f"{TH}/raw/{name}"; os.makedirs(out)
    cmd = ["timeout", str(timeout), "codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
           "-m", "gpt-6-astra", "-s", "read-only", "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="high"',
           "-c", 'web_search="disabled"', "--output-schema", schema_path, "--json", "-o", f"{out}/last_message.json", "-C", ws, "--color", "never"]
    if images: cmd += ["-i", *images]
    env = {**os.environ, "PATH": f"{H}/langres_state/venv/bin:" + os.environ["PATH"]} if agent else None
    t0 = time.time(); p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=ws, env=env); lat = round(time.time() - t0, 1)
    open(f"{out}/events.jsonl", "w").write(p.stdout); open(f"{out}/stderr.txt", "w").write(p.stderr)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    cmds = [e["item"] for e in ev if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "command_execution"]
    json.dump(cmds, open(f"{out}/commands.json", "w"), indent=1)
    items = sorted({e["item"]["type"] for e in ev if "item" in e})
    flags = (audit(ws, [c.get("command", "") for c in cmds]) if agent else [("tool", t, "") for t in items if t in TOOL_ITEMS])
    flags += [("tool", t, "") for t in items if agent and t in ("web_search", "mcp_tool_call", "file_change")]
    limit = p.returncode != 0 and any(w in (p.stdout + p.stderr).lower() for w in LIMIT_WORDS)
    rec = {"name": name, "returncode": p.returncode, "latency_s": lat, "n_commands": len(cmds), "item_types": items, "audit_flags": flags,
           "usage_limit": limit, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "cmd": " ".join(cmd[:-len(images) - 1] if images else cmd),
           "n_images": len(images), "raw_dir": os.path.relpath(out, TH), "usage": [e.get("usage") for e in ev if e.get("type") == "turn.completed"],
           "t_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return rec, out


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


FORBID = {"open", "eval", "exec", "__import__", "compile", "globals", "locals", "getattr", "setattr", "input", "breakpoint", "os", "sys", "subprocess", "socket"}


def check_code(code):
    """Design check by AST: imports only numpy, scipy, math; none of the forbidden names or attributes; a top-level label(x)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax: {e}"]
    errs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for m in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                if m.split(".")[0] not in ("numpy", "scipy", "math"): errs.append(f"import {m}")
        if isinstance(n, ast.Name) and n.id in FORBID: errs.append(f"name {n.id}")
        if isinstance(n, ast.Attribute) and n.attr in FORBID: errs.append(f"attr {n.attr}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "label" for n in tree.body): errs.append("no top-level label(x)")
    return sorted(set(errs))


def ba(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    return None if y.all() or (~y).all() else 0.5 * (p[y].mean() + (~p[~y]).mean())


def boot(y, p, pair=None, n=1000, seed=0):
    """95% bootstrap CI over patches of BA(p), or of BA(p) - BA(pair)."""
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); a = ba(y[i], p[i])
        if a is not None: v.append(a - ba(y[i], pair[i]) if pair is not None else a)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
