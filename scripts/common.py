"""Shared constants, prompts and the frozen Qwen2.5-VL scorer (pass 1 -> z, pass 2 -> z_hat)."""
import os, json
import numpy as np
from PIL import Image

HOME = os.path.expanduser("~")
ARR = f"{HOME}/langres_data/arrays"
CAL = f"{HOME}/langres_data/calib"
STATE = f"{HOME}/langres_state"
REPO = f"{HOME}/Documents/langres"
REVS = json.load(open(f"{REPO}/model_revisions.json"))
VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
LAYER = 21  # hidden_states[21] = output of decoder block 21 of 28 (LR-X16)
IMG_SIDE = 448

PREAMBLE_B = "The second image is a false color composite of the same patch with near infrared shown as red."
DESCRIBE = ("Describe this Sentinel-2 satellite image patch for a scientist in at most 100 words. "
            "Cover land cover, vegetation, water, built structures, texture and spatial layout.")
Q = "What is the key content of this Sentinel-2 patch?"
PREFILL = "The key content of this patch is"
REWRITE_HEAD = "Here are earlier descriptions of this patch with their scores. Higher is better."
REWRITE_TAIL = ("Write one new description in at most 100 words that states what the image shows more completely "
                "and precisely. Do not mention scores.")
B0_TEXT = "A satellite image patch."


def rev(m):
    return REVS[m]["revision"]


def ids(split):
    return json.load(open(f"{ARR}/{split}_ids.json"))


def n_words(w):
    return len(w.split())


def to_img(a):
    return Image.fromarray(np.asarray(a)).resize((IMG_SIDE, IMG_SIDE), Image.BICUBIC)


def arm_images(arm, rgb, fcc):
    return [to_img(rgb)] + ([to_img(fcc)] if arm == "B" else [])


def with_preamble(arm, text):
    # LR-X17: preamble and instruction joined by one space in a single text block after the images
    return (PREAMBLE_B + " " + text) if arm == "B" else text


def image_user(arm, text):
    n = 2 if arm == "B" else 1
    return {"role": "user", "content": [{"type": "image"}] * n + [{"type": "text", "text": with_preamble(arm, text)}]}


def rewrite_text(top4):
    # LR-X18: top 4 listed best first as "Description k (score x.xxx): w", blank line separated
    body = "\n\n".join(f"Description {k + 1} (score {s:.3f}): {w}" for k, (w, s) in enumerate(top4))
    return f"{REWRITE_HEAD}\n\n{body}\n\n{REWRITE_TAIL}"


def pass2_user(w):
    return {"role": "user", "content": [{"type": "text", "text": "Here is a description of a Sentinel-2 patch: " + w + " " + Q}]}


class QwenScorer:
    """Frozen Qwen2.5-VL forward passes returning the layer-21 state of the last prefilled token."""

    def __init__(self, model=None, proc=None, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.proc = proc or AutoProcessor.from_pretrained(VLM, revision=rev(VLM))
        self.proc.tokenizer.padding_side = "left"
        self.model = model or Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM, revision=rev(VLM), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device = device

    def _prompt(self, user):
        return self.proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True) + PREFILL

    def _run(self, texts, images=None):
        torch = self.torch
        kw = dict(text=texts, padding=True, return_tensors="pt")
        if images is not None:
            kw.update(images=images, images_kwargs={"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE})
        enc = self.proc(**kw).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
        return out.hidden_states[LAYER][:, -1].float().cpu().numpy()

    def pass1(self, arm, img_lists, bs=16):
        res = []
        for k in range(0, len(img_lists), bs):
            chunk = img_lists[k:k + bs]
            texts = [self._prompt(image_user(arm, Q))] * len(chunk)
            res.append(self._run(texts, [im for ims in chunk for im in ims]))
        return np.concatenate(res)

    def pass2(self, summaries, bs=64):
        res = []
        for k in range(0, len(summaries), bs):
            res.append(self._run([self._prompt(pass2_user(w)) for w in summaries[k:k + bs]]))
        return np.concatenate(res)


def unit(x, mu):
    d = x - mu
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def load_means(arm):
    m = np.load(f"{CAL}/means_{arm}.npz")
    return m["mu_img"], m["mu_txt"]


def evict_model_cache(repo):
    """Drop page cache of our own HF blobs for `repo` (GB10 unified memory: cache competes with CUDA allocations)."""
    import glob
    for p in glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/blobs/*")):
        try:
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass
