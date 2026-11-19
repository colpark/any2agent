"""Shared FN helpers. Reuses teacher_heads/scripts/th_common.py (pilot-style renderings, Codex wrapper with the image-QA
settings and audit) by pointing its output folders at fm_names."""
import json, os, sys
import numpy as np
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; FN = f"{REPO}/fm_names"; ST = f"{H}/langres_state/fn"
sys.path.insert(0, f"{REPO}/teacher_heads/scripts")
import th_common as thc
thc.TH, thc.ST = FN, ST  # codex_call raw dirs -> fm_names/raw, renders -> ~/langres_state/fn/renders
from th_common import ARR, PR, codex_call
os.makedirs(ST, exist_ok=True); LEDGER = f"{FN}/calls.jsonl"; CAP = 24


def composites(split, idx):
    """One 1,024 x 512 composite per patch: stage-1 true color (left) + false color (right), random file names."""
    from PIL import Image
    halves = thc.render(split, idx); out = {}
    for i, (a, b) in halves.items():
        p = a[:-5] + ".png"; im = Image.new("RGB", (1024, 512)); im.paste(Image.open(a), (0, 0)); im.paste(Image.open(b), (512, 0)); im.save(p)
        os.remove(a); os.remove(b); out[i] = p
    return out


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]
