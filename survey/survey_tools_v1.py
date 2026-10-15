#!/usr/bin/env python3
"""
survey_tools.py  (version 2)

Reproducibility tools for the survey
"From Seeing to Acting: A Survey of Vision-Language and Vision-Language-Action
Models for Autonomous Driving".

Changes with respect to version 1
  * Strict answer parsing for reasoning outputs (no last-line guessing).
    A response without a valid label is "unparsed" and is scored incorrect.
  * Truncation (hitting max_new_tokens) is recorded per response.
    The default reasoning cap is now 512 tokens.
  * make-tables re-parses every stored raw response with the current parser,
    so results.jsonl files from version 1 can be re-scored without re-running.
  * Scene-clustered bootstrap CIs, balanced accuracy, yes-rate, unparsed and
    truncation rates, automatic flags for degenerate (near-constant) outputs.
  * E2: paired accuracy differences with CIs, latency fit with intercept,
    p50 / p95 / p99 / max, warm-up for both prompts.
  * Provenance: script hash, git commit, model revision, preprocessing
    settings, prompts.json.
  * merge: replace selected (model, experiment, condition) blocks of a run
    with a re-run, without touching the rest.
  * summary, check-links, screen-kappa; kappa now reports CIs and confusion
    matrices; PRISMA search writes a search log.

Subcommands
  drivelm-stats    Recompute the DriveLM target-action distribution (Fig. 9).
  casestudy        Section 7. Runs E1 (grounding control) and E2 (latency cost
                   of reasoning); writes results.jsonl, items.json, meta.json,
                   prompts.json.
  merge            Replace blocks of a results file with a re-run.
  make-tables      Section 7. Builds tables, setup text, draft sentences,
                   ci.csv, e2_stats.csv, diagnostics.txt.
  summary          Section 3. Counts systems by coupling, reasoning and year
                   (checks the numbers quoted in the text).
  kappa-template   Section 3.4. Blank annotation sheets for two annotators.
  kappa            Section 3.4. Cohen's kappa with CIs and confusion matrices.
  prisma-search    Section 1.2. arXiv search + search log + screening sheet.
  screen-kappa     Section 1.2. Cohen's kappa for two screening decisions.
  prisma-count     Section 1.2. PRISMA counts and supplementary Table S1.
  check-links      Print the arXiv title for every arXiv id in the system list.
  readme           Conclusion. README.md for the companion repository.

Requirements
  All subcommands : Python 3.9+
  casestudy       : pip install "torch>=2.3" "transformers>=4.53" accelerate pillow numpy
                    One CUDA GPU with at least 24 GB of memory.
  make-tables, merge, kappa, summary : pip install numpy

Typical order
  python survey_tools.py drivelm-stats --drivelm-json v1_1_train_nus.json
  python survey_tools.py casestudy ... --outdir cs_out
  python survey_tools.py make-tables --outdir cs_out
  (re-run E2 with a larger cap, then:)
  python survey_tools.py merge --base cs_out --rerun cs_rerun --outdir cs_final
  python survey_tools.py make-tables --outdir cs_final
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:  # only the GPU / table commands need numpy
    np = None

SCRIPT_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
VIEW_W, VIEW_H = 448, 252          # per-view size in the 2x3 tiled image
ACTIONS = ["go straight", "stationary", "turn left", "turn right"]
YESNO = ["yes", "no"]
CONDITIONS = ["original", "blank", "corrupted", "noimage"]

DEFAULT_MAX_DIRECT = 64
DEFAULT_MAX_REASON = 512
# Token caps assumed for records that carry no "truncated" flag (files written by
# version 1 of this script). Override with --legacy-cap-direct / --legacy-cap-reason.
LEGACY = {"direct": 16, "reason": 256}

# key -> (Hugging Face model id, label used in the paper)
MODELS = {
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
    "internvl3-8b": ("OpenGVLab/InternVL3-8B-hf", "InternVL3-8B"),
    "llava-1.5-7b": ("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5-7B"),
}

# Table IV of the survey: (system, venue, coupling, reasoning, arXiv id or "")
# VERIFICATION REQUIRED: every venue, year and arXiv id below must be checked
# against the primary publication (see the check-links subcommand).
SYSTEMS = [
    ("Talk2BEV", "ICRA 2024", "I", "R0", ""),
    ("BEV-InMLLM", "CVPR 2024", "I", "R0", ""),
    ("Dolphins", "ECCV 2024", "I", "R1", ""),
    ("EM-VLM4AD", "CVPRW 2024", "I", "R0", ""),
    ("Reason2Drive", "ECCV 2024", "I", "R2", ""),
    ("TS-VLM", "arXiv 2025", "I", "R0", "2505.12670"),
    ("DriveLMM-o1", "arXiv 2025", "I", "R1", "2503.10621"),
    ("OmniDrive-R1", "arXiv 2025", "I", "R3", "2512.14044"),
    ("LanguageMPC", "arXiv 2023", "M", "R1", "2310.03026"),
    ("DriveMLM", "arXiv 2023", "M", "R1", "2312.09245"),
    ("DiLu", "ICLR 2024", "M", "R4", ""),
    ("Senna", "arXiv 2024", "M", "R1", "2410.22313"),
    ("AlphaDrive", "arXiv 2025", "M", "R1", "2503.07608"),
    ("DriveVLM-Dual", "CoRL 2024", "D", "R1", ""),
    ("LeapAD", "NeurIPS 2024", "D", "R4", ""),
    ("ETA", "ICCV 2025", "D", "R0", ""),
    ("GPT-Driver", "arXiv 2023", "U", "R1", "2310.01415"),
    ("Agent-Driver", "arXiv 2023", "U", "R4", "2311.10813"),
    ("EMMA", "arXiv 2024", "U", "R1", "2410.23262"),
    ("DriveGPT4", "RA-L 2024", "U", "R1", "2310.01412"),
    ("Driving with LLMs", "ICRA 2024", "U", "R1", ""),
    ("RAG-Driver", "RSS 2024", "U", "R1", "2402.10828"),
    ("DriveVLM", "CoRL 2024", "U", "R1", ""),
    ("DriveLM-Agent", "ECCV 2024", "U", "R2", ""),
    ("LMDrive", "CVPR 2024", "U", "R0", ""),
    ("DriveCoT", "arXiv 2024", "U", "R1", "2403.16996"),
    ("RDA-Driver", "ECCV 2024", "U", "R1", ""),
    ("OccLLaMA", "arXiv 2024", "U", "R0", "2409.03272"),
    ("Doe-1", "arXiv 2024", "U", "R1", "2412.09627"),
    ("BEVDriver", "IROS 2025", "U", "R0", ""),
    ("OmniDrive", "CVPR 2025", "U", "R1", ""),
    ("SimLingo", "CVPR 2025", "U", "R1", ""),
    ("ORION", "ICCV 2025", "U", "R1", ""),
    ("DriveMoE", "arXiv 2025", "U", "R0", "2505.16278"),
    ("Alpamayo-R1", "arXiv 2025", "U", "R1", "2511.00088"),
    ("AutoVLA", "NeurIPS 2025", "U", "R4", "2506.13757"),
    ("AdaThinkDrive", "arXiv 2025", "U", "R4", "2509.13769"),
    ("WAM-Diff", "arXiv 2025", "U", "R0", "2512.11872"),
    ("OpenDriveVLA", "AAAI 2026", "U", "R0", "2503.23463"),
    ("AutoDrive-R2", "ICLR 2026", "U", "R4", "2509.01944"),
    ("ColaVLA", "CVPR 2026", "U", "R3", "2512.22939"),
    ("DriveVLA-W0", "ICLR 2026", "U", "R0", "2510.12796"),
    ("DriveFine", "ECCV 2026", "U", "R0", "2602.14577"),
]

# Systems that use reinforcement-learning post-training. Left EMPTY on purpose:
# fill it from the primary publications (do not guess), then run `summary` to
# check the sentence "N systems use RL post-training, all first appeared in
# 2025 or 2026".
RL_POSTTRAINED = set()

COUPLING_NAMES = {"I": "Interpretive", "M": "Language-mediated",
                  "D": "Dual-system", "U": "Unified"}
REASONING_NAMES = {"R0": "Direct", "R1": "Textual", "R2": "Structured",
                   "R3": "Non-verbal", "R4": "Adaptive or reflective"}

DEFINITIONS = """CLASS DEFINITIONS (Section 3 of the survey)

Coupling axis (how language is coupled to action at inference)
  I  Interpretive: f: V x T -> T. Output text is not consumed by any
     component that sets actuator commands.
  M  Language-mediated: g(f(.)), f outputs a discrete language decision
     (meta-action, behavior state, program); g is a separately designed
     planner or controller, not trained end to end with f.
  D  Dual-system: a slow model updates guidance at a low rate; a fast
     planner acts every control cycle using the latest guidance.
  U  Unified: one network trained end to end outputs actions (text
     waypoints, action tokens, a learned head, or a generative decoder
     conditioned on the language backbone).

Reasoning axis (what the model produces before its final output)
  R0 Direct: no intermediate reasoning variable.
  R1 Textual: free-form text such as a chain of thought or explanation.
  R2 Structured: reasoning in a fixed schema, e.g., a graph of QA pairs.
  R3 Non-verbal: latent vectors or visual artifacts (crops, frames).
  R4 Adaptive or reflective: the presence or length of reasoning is a
     learned decision, or reasoning contains a self-evaluation step that
     can revise a candidate output.

Instructions: classify each system from its primary publication only.
Do not consult the survey's Table IV or discuss with the other annotator.
"""

# ---------------------------------------------------------------------------
# Answer normalization (shared by drivelm-stats, casestudy, make-tables)
# ---------------------------------------------------------------------------

_STAT = re.compile(r"\b(stationary|stop|stops|stopped|stay|stays|remain|remains|"
                   r"halt|wait|waits|parked|standstill|not moving|keep still)\b")
_LEFT = re.compile(r"\bleft\b")
_RIGHT = re.compile(r"\bright\b")
_STRAIGHT = re.compile(r"\b(straight|forward|ahead)\b")
_YN_START = re.compile(r"^\s*(yes|no)\b", re.I)
_YN_ANY = re.compile(r"\b(yes|no)\b", re.I)
_ANSWER = re.compile(r"answer\s*:\s*[\*_`\"' ]*([^\n]+)", re.I)


def action_category(text):
    """Map a free-text action to one of ACTIONS or 'other'.
    If drivelm-stats does not reproduce 72.2 / 16.7 / 5.2 / 5.1 / 0.8,
    adjust the regular expressions above."""
    t = (text or "").lower()
    if _STAT.search(t):
        return "stationary"
    left, right = bool(_LEFT.search(t)), bool(_RIGHT.search(t))
    if left and not right:
        return "turn left"
    if right and not left:
        return "turn right"
    if _STRAIGHT.search(t):
        return "go straight"
    return "other"


def parse_yesno(text):
    m = _YN_ANY.search(text or "")
    return m.group(1).lower() if m else "other"


def parse_action(text, reasoning):
    """Strict parser. For reasoning outputs the label is read ONLY from the
    last 'Answer: <option>' marker; a response without the marker (for
    example because it was truncated) is 'unparsed' and counts as incorrect.
    Version 1 fell back to the last line, which turned half-finished
    sentences into predictions."""
    if reasoning:
        hits = _ANSWER.findall(text or "")
        return action_category(hits[-1]) if hits else "unparsed"
    return action_category(text)


def reparse(rec):
    """Re-derive the prediction from the stored raw response."""
    raw = rec.get("raw", "")
    if rec["task"] == "yesno":
        return parse_yesno(raw)
    return parse_action(raw, rec["exp"] == "e2" and rec["cond"] == "reasoning")


# ---------------------------------------------------------------------------
# DriveLM loading
# ---------------------------------------------------------------------------

def load_drivelm(path):
    """Return a list of (scene_id, frame_id, frame_dict).
    Expected layout: {scene: {"key_frames": {frame: {"QA": {...},
    "image_paths": {CAM: path}}}}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for scene_id, scene in data.items():
        if not isinstance(scene, dict):
            continue
        for frame_id, fr in (scene.get("key_frames") or {}).items():
            frames.append((scene_id, frame_id, fr))
    if not frames:
        sys.exit("No key frames found. Check that --drivelm-json points to "
                 "v1_1_train_nus.json and that its layout matches load_drivelm().")
    return frames


def iter_qa(frame):
    for cat, items in (frame.get("QA") or {}).items():
        for i, it in enumerate(items or []):
            yield cat, i, str(it.get("Q", "")), str(it.get("A", ""))


def collect_items(frames):
    """Split QA items into target-action items and yes/no items."""
    action, yesno = [], []
    for scene_id, frame_id, fr in frames:
        images = fr.get("image_paths") or {}
        for cat, i, q, a in iter_qa(fr):
            rec = {"uid": "%s:%s:%d" % (frame_id, cat, i), "scene": scene_id,
                   "frame": frame_id, "cat": cat, "q": q, "raw_answer": a,
                   "images": images}
            if cat == "planning" and "target action" in q.lower():
                rec["gt"] = action_category(a)
                action.append(rec)
            elif _YN_START.match(a):
                rec["gt"] = _YN_START.match(a).group(1).lower()
                yesno.append(rec)
    return action, yesno


def cmd_drivelm_stats(args):
    frames = load_drivelm(args.drivelm_json)
    action, yesno = collect_items(frames)
    n = len(action)
    counts = Counter(r["gt"] for r in action)
    print("Key frames: %d" % len(frames))
    print("Target-action items (planning, question contains 'target action'): %d" % n)
    expected = {"go straight": 72.2, "stationary": 16.7, "turn right": 5.2,
                "turn left": 5.1, "other": 0.8}
    print("\n%-12s %8s %8s %10s" % ("class", "count", "share", "expected"))
    for c in ["go straight", "stationary", "turn right", "turn left", "other"]:
        share = 100.0 * counts[c] / n if n else 0.0
        print("%-12s %8d %7.1f%% %9.1f%%" % (c, counts[c], share, expected[c]))
    four = sum(counts[c] for c in ACTIONS)
    if four:
        print("\nShare of 'go straight' after dropping 'other' (four-class pool "
              "used in the case study): %.1f%%" % (100.0 * counts["go straight"] / four))
    print("\nMost frequent raw answers per class (check the mapping):")
    raw = defaultdict(Counter)
    for r in action:
        raw[r["gt"]][r["raw_answer"].strip()] += 1
    for c, ctr in raw.items():
        print("  [%s]" % c)
        for s, k in ctr.most_common(args.top):
            print("     %5d  %s" % (k, s[:100]))
    yn = Counter(r["gt"] for r in yesno)
    print("\nYes/no items available: %d (yes %d, no %d)" % (len(yesno), yn["yes"], yn["no"]))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def resolve_image(path, roots):
    if not path:
        return None
    p = str(path).replace("\\", "/")
    tail = p.split("samples/", 1)[1] if "samples/" in p else os.path.basename(p)
    for r in roots:
        for c in (os.path.join(r, "samples", tail), os.path.join(r, tail),
                  os.path.join(r, os.path.basename(p))):
            if os.path.isfile(c):
                return c
    return None


def has_all_images(item, roots):
    return all(resolve_image(item["images"].get(c, ""), roots) for c in CAMERAS)


_TILE_CACHE = {}


def tiled_image(item, roots):
    from PIL import Image
    key = item["frame"]
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]
    canvas = Image.new("RGB", (3 * VIEW_W, 2 * VIEW_H), (0, 0, 0))
    for k, cam in enumerate(CAMERAS):
        p = resolve_image(item["images"].get(cam, ""), roots)
        im = Image.open(p).convert("RGB").resize((VIEW_W, VIEW_H), Image.BICUBIC)
        canvas.paste(im, ((k % 3) * VIEW_W, (k // 3) * VIEW_H))
    if len(_TILE_CACHE) > 256:
        _TILE_CACHE.clear()
    _TILE_CACHE[key] = canvas
    return canvas


def pad_to_square(img, fill=(0, 0, 0)):
    from PIL import Image
    s = max(img.size)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - img.size[0]) // 2, (s - img.size[1]) // 2))
    return out


def apply_condition(img, cond, uid):
    from PIL import Image, ImageFilter
    import numpy as _np
    if cond == "original":
        return img
    if cond == "blank":
        return Image.new("RGB", img.size, (128, 128, 128))
    if cond == "corrupted":
        rng = _np.random.default_rng(zlib.crc32(uid.encode("utf-8")))
        arr = _np.asarray(img).astype(_np.float32)
        arr = _np.clip(arr + rng.normal(0.0, 25.0, arr.shape), 0, 255).astype(_np.uint8)
        return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=2))
    if cond == "noimage":
        return None
    raise ValueError(cond)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONTEXT = ("You see the six surround-view camera images of an ego vehicle, tiled in two rows. "
           "Top row, left to right: front-left, front, front-right. "
           "Bottom row, left to right: back-left, back, back-right. "
           "Objects may be referenced as <id,camera,x,y>, where x and y are pixel "
           "coordinates in the original 1600x900 image of that camera.")


def prompt_yesno(q):
    return "%s\n\nQuestion: %s\nAnswer with a single word, yes or no." % (CONTEXT, q)


def prompt_action_direct(q):
    return ("%s\n\nQuestion: %s\nChoose exactly one option: go straight, stationary, "
            "turn left, turn right. Reply with the option only." % (CONTEXT, q))


def prompt_action_reason(q):
    return ("%s\n\nQuestion: %s\nThink step by step about the relevant objects, traffic "
            "rules, and the situation of the ego vehicle in a few sentences. Then write the "
            "final line as 'Answer: <option>', where <option> is one of: go straight, "
            "stationary, turn left, turn right." % (CONTEXT, q))


def prompt_templates():
    """Prompt templates for the supplementary material (placeholder QUESTION)."""
    return {"context": CONTEXT,
            "yesno": prompt_yesno("QUESTION"),
            "action_direct": prompt_action_direct("QUESTION"),
            "action_reasoning": prompt_action_reason("QUESTION")}


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def script_sha256():
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class VLM:
    def __init__(self, key, attn=None, pad_square=False):
        import torch
        import transformers
        from transformers import AutoProcessor
        if not torch.cuda.is_available():
            sys.exit("casestudy needs a CUDA GPU.")
        self.torch = torch
        self.key = key
        self.pad_square = bool(pad_square) and key == "llava-1.5-7b"
        mid = MODELS[key][0]
        self.processor = AutoProcessor.from_pretrained(mid)
        major = int(transformers.__version__.split(".")[0])
        kw = {("dtype" if major >= 5 else "torch_dtype"): torch.bfloat16,
              "device_map": "cuda"}
        if attn:
            kw["attn_implementation"] = attn
        try:
            from transformers import AutoModelForImageTextToText as AutoCls
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoCls
        self.model = AutoCls.from_pretrained(mid, **kw).eval()
        ip = getattr(self.processor, "image_processor", None)
        pre = {}
        for k in ("do_resize", "size", "do_center_crop", "crop_size", "do_pad",
                  "min_pixels", "max_pixels", "max_patches", "do_convert_rgb"):
            if ip is not None and hasattr(ip, k):
                pre[k] = getattr(ip, k)
        self.env = {"model_id": mid,
                    "model_revision": getattr(self.model.config, "_commit_hash", None),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "dtype": "bfloat16",
                    "attn_implementation": attn or "default",
                    "image_processor": pre,
                    "pad_square": self.pad_square,
                    "script_version": SCRIPT_VERSION,
                    "script_sha256": script_sha256(),
                    "git_commit": git_commit()}
        print("  image_processor settings: %s" % json.dumps(pre, default=str))

    def generate(self, prompt, image, max_new_tokens):
        torch = self.torch
        if image is not None and self.pad_square:
            image = pad_to_square(image)
        content = ([{"type": "image"}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if image is not None:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[text], return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        gen = out[:, n_in:]
        txt = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        ntok = int(gen.shape[1])
        info = {"n_input_tokens": n_in, "truncated": bool(ntok >= max_new_tokens),
                "max_new_tokens": int(max_new_tokens)}
        return txt.strip(), ntok, ms, info

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Case study
# ---------------------------------------------------------------------------

def sample_items(json_path, roots, n_yn, n_act, n_e2, seed):
    frames = load_drivelm(json_path)
    action, yesno = collect_items(frames)
    action = [a for a in action if a["gt"] in ACTIONS]
    rng = random.Random(seed)
    rng.shuffle(action)
    rng.shuffle(yesno)

    def take(pool, n):
        out, missing = [], 0
        for it in pool:
            if len(out) >= n:
                break
            if has_all_images(it, roots):
                out.append(it)
            else:
                missing += 1
        return out, missing

    act_s, m1 = take(action, n_act)
    yn_s, m2 = take(yesno, n_yn)
    if m1 + m2 > 0:
        print("Skipped %d items with missing images." % (m1 + m2))
    if not act_s and not yn_s:
        sys.exit("No items with all six images found. Check --image-root.")
    if len(act_s) < n_act or len(yn_s) < n_yn:
        print("Warning: fewer items than requested (action %d, yes/no %d)." % (len(act_s), len(yn_s)))
    return {"seed": seed, "yesno": yn_s, "action": act_s, "e2": act_s[:n_e2],
            "pool_sizes": {"action_4class": len(action), "yesno": len(yesno)},
            "skipped_missing_images": m1 + m2}


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["exp"], r["cond"], r["uid"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def cmd_casestudy(args):
    os.makedirs(args.outdir, exist_ok=True)
    items_path = os.path.join(args.outdir, "items.json")
    results_path = os.path.join(args.outdir, "results.jsonl")
    meta_path = os.path.join(args.outdir, "meta.json")
    roots = args.image_root

    if os.path.exists(items_path):
        with open(items_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        print("Reusing sampled items from %s" % items_path)
    else:
        items = sample_items(args.drivelm_json, roots, args.n_yesno, args.n_action,
                             args.n_e2, args.seed)
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump(items, f)
    print("Items: yes/no %d, target-action %d, E2 %d" %
          (len(items["yesno"]), len(items["action"]), len(items["e2"])))
    with open(os.path.join(args.outdir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_templates(), f, indent=1)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    done = load_done(results_path)
    exps = [e.strip() for e in args.experiments.split(",")]
    model_keys = [m.strip() for m in args.models.split(",")]
    md, mr = args.max_new_tokens_direct, args.max_new_tokens_reason

    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    with open(results_path, "a", encoding="utf-8") as fout:
        for mkey in model_keys:
            if mkey not in MODELS:
                sys.exit("Unknown model key %s. Choose from %s" % (mkey, ", ".join(MODELS)))
            print("\n=== Loading %s ===" % MODELS[mkey][0])
            vlm = VLM(mkey, args.attn, pad_square=args.llava_pad_square)
            meta[mkey] = vlm.env
            meta[mkey]["max_new_tokens_direct"] = md
            meta[mkey]["max_new_tokens_reason"] = mr
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1, default=str)

            def write(rec):
                fout.write(json.dumps(rec) + "\n")
                fout.flush()

            if "e1" in exps:
                for task, pool in (("yesno", items["yesno"]), ("action", items["action"])):
                    for n_it, it in enumerate(pool):
                        base = tiled_image(it, roots)
                        for cond in CONDITIONS:
                            key = (mkey, "e1", cond, it["uid"])
                            if key in done:
                                continue
                            img = apply_condition(base, cond, it["uid"])
                            prompt = prompt_yesno(it["q"]) if task == "yesno" else prompt_action_direct(it["q"])
                            raw, ntok, ms, info = vlm.generate(prompt, img, md)
                            pred = parse_yesno(raw) if task == "yesno" else parse_action(raw, False)
                            rec = {"model": mkey, "exp": "e1", "cond": cond, "task": task,
                                   "uid": it["uid"], "gt": it["gt"], "raw": raw, "pred": pred,
                                   "correct": int(pred == it["gt"]), "tokens": ntok, "ms": ms}
                            rec.update(info)
                            write(rec)
                            done.add(key)
                        if (n_it + 1) % 50 == 0:
                            print("  E1 %s %s: %d/%d items" % (mkey, task, n_it + 1, len(pool)))

            if "e2" in exps and items["e2"]:
                first = items["e2"][0]
                img0 = tiled_image(first, roots)
                for _ in range(3):  # warm-up for both prompts, not recorded
                    vlm.generate(prompt_action_direct(first["q"]), img0, md)
                    vlm.generate(prompt_action_reason(first["q"]), img0, mr)
                for mode in ("direct", "reasoning"):
                    for n_it, it in enumerate(items["e2"]):
                        key = (mkey, "e2", mode, it["uid"])
                        if key in done:
                            continue
                        img = tiled_image(it, roots)
                        if mode == "direct":
                            raw, ntok, ms, info = vlm.generate(prompt_action_direct(it["q"]), img, md)
                        else:
                            raw, ntok, ms, info = vlm.generate(prompt_action_reason(it["q"]), img, mr)
                        pred = parse_action(raw, mode == "reasoning")
                        rec = {"model": mkey, "exp": "e2", "cond": mode, "task": "action",
                               "uid": it["uid"], "gt": it["gt"], "raw": raw, "pred": pred,
                               "correct": int(pred == it["gt"]), "tokens": ntok, "ms": ms}
                        rec.update(info)
                        write(rec)
                        done.add(key)
                        if (n_it + 1) % 50 == 0:
                            print("  E2 %s %s: %d/%d items" % (mkey, mode, n_it + 1, len(items["e2"])))
            vlm.close()
    print("\nDone. Results in %s. Next: python survey_tools.py make-tables --outdir %s"
          % (results_path, args.outdir))


def cmd_merge(args):
    """Replace every (model, exp, cond) block present in --rerun by the re-run rows."""
    def read(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    base = read(os.path.join(args.base, "results.jsonl"))
    rerun = read(os.path.join(args.rerun, "results.jsonl"))
    if not rerun:
        sys.exit("No rows in the re-run file.")
    with open(os.path.join(args.base, "items.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    base_uids = {(r["exp"], r["uid"]) for r in base}
    unknown = [r["uid"] for r in rerun if (r["exp"], r["uid"]) not in base_uids]
    if unknown:
        sys.exit("Re-run contains %d items that are not in the base run (different sample?). "
                 "Copy items.json from the base run into the re-run directory before running." % len(unknown))
    blocks = {(r["model"], r["exp"], r["cond"]) for r in rerun}
    kept = [r for r in base if (r["model"], r["exp"], r["cond"]) not in blocks]
    # Report how many answers changed between base and re-run (replication check).
    old = {(r["model"], r["exp"], r["cond"], r["uid"]): r for r in base}
    changed = Counter()
    total = Counter()
    for r in rerun:
        k = (r["model"], r["exp"], r["cond"], r["uid"])
        if k in old:
            total[k[:3]] += 1
            if old[k].get("raw") != r.get("raw"):
                changed[k[:3]] += 1
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in kept + rerun:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.outdir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f)
    meta = {}
    for d in (args.base, args.rerun):
        p = os.path.join(d, "meta.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    meta.setdefault(k, {}).update(v)
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    for fn in ("prompts.json",):
        for d in (args.rerun, args.base):
            if os.path.exists(os.path.join(d, fn)):
                shutil.copy(os.path.join(d, fn), os.path.join(args.outdir, fn))
                break
    print("Merged: %d base rows kept, %d re-run rows added -> %s" %
          (len(kept), len(rerun), os.path.join(args.outdir, "results.jsonl")))
    print("Replication check (identical raw responses base vs re-run; differences are expected "
          "only where the old run was truncated):")
    for k in sorted(total):
        print("  %-16s %-3s %-10s changed %d / %d" % (k + (changed[k], total[k])))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def fmt(x, nd=1):
    return "--" if x is None else "%.*f" % (nd, x)


def span(vals, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    lo, hi = min(vals), max(vals)
    return fmt(lo, nd) if abs(hi - lo) < 10 ** (-nd) else "%s to %s" % (fmt(lo, nd), fmt(hi, nd))


def signed(x, nd=1):
    s = fmt(x, nd)
    return s.replace("-", "$-$") if s != "--" else s


def cluster_ci(vals, scene_of, B=2000, seed=0):
    """Mean of per-item values with a scene-clustered bootstrap 95% CI (in %)."""
    groups = defaultdict(list)
    for u, v in vals.items():
        groups[scene_of.get(u, u)].append(v)
    if not groups:
        return None, None, None
    sums = np.array([sum(g) for g in groups.values()], dtype=float)
    cnt = np.array([len(g) for g in groups.values()], dtype=float)
    est = sums.sum() / cnt.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (B, len(sums)))
    boots = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return 100.0 * est, 100.0 * np.percentile(boots, 2.5), 100.0 * np.percentile(boots, 97.5)


def balanced_acc(rs):
    rec = []
    for c in ACTIONS:
        g = [r for r in rs if r["gt"] == c]
        if g:
            rec.append(np.mean([r["pred"] == c for r in g]))
    return 100.0 * float(np.mean(rec)) if rec else None


def top_share(rs):
    if not rs:
        return None
    c = Counter(r["pred"] for r in rs)
    return max(c.values()) / len(rs)


def unparsed_rate(rs, task):
    valid = YESNO if task == "yesno" else ACTIONS
    return sum(r["pred"] not in valid for r in rs) / len(rs) if rs else None


def lenient_pred(r):
    """Version-1 style prediction for reasoning outputs: last 'Answer:' marker, else the
    last non-empty line. Reported ONLY as a sensitivity number next to the strict one."""
    raw = r.get("raw", "")
    hits = _ANSWER.findall(raw)
    if hits:
        return action_category(hits[-1])
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    return action_category(lines[-1]) if lines else "other"


def is_truncated(r):
    t = r.get("truncated")
    if t is not None:
        return bool(t)
    cap = LEGACY["reason"] if (r["exp"] == "e2" and r["cond"] == "reasoning") else LEGACY["direct"]
    return r["tokens"] >= cap          # version-1 files carry no flag


# ---------------------------------------------------------------------------
# Tables and text for Section 7
# ---------------------------------------------------------------------------

def cmd_make_tables(args):
    if np is None:
        sys.exit("make-tables needs numpy.")
    if args.legacy_cap_direct:
        LEGACY["direct"] = args.legacy_cap_direct
    if args.legacy_cap_reason:
        LEGACY["reason"] = args.legacy_cap_reason
    d = args.outdir
    with open(os.path.join(d, "items.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    scene_of = {it["uid"]: it["scene"] for it in items["yesno"] + items["action"]}

    recs = {}
    with open(os.path.join(d, "results.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["pred"] = reparse(r)                      # strict, current parser
            r["correct"] = int(r["pred"] == r["gt"])
            recs[(r["model"], r["exp"], r["cond"], r["uid"])] = r
    models = [m for m in MODELS if any(k[0] == m for k in recs)]
    if not models:
        sys.exit("No results found in results.jsonl.")

    def rows(m, exp, cond, pool):
        return [recs[(m, exp, cond, it["uid"])] for it in pool if (m, exp, cond, it["uid"]) in recs]

    def acc(rs):
        return 100.0 * float(np.mean([r["correct"] for r in rs])) if rs else None

    pools = {"yesno": items["yesno"], "action": items["action"]}
    n_yn, n_act, n_e2 = len(items["yesno"]), len(items["action"]), len(items["e2"])
    gts_yn = Counter(it["gt"] for it in items["yesno"])
    maj_label, maj_n = gts_yn.most_common(1)[0]
    maj = 100.0 * maj_n / n_yn
    true_yes = 100.0 * gts_yn["yes"] / n_yn
    const = 100.0 * sum(it["gt"] == "go straight" for it in items["action"]) / n_act
    const_e2 = 100.0 * sum(it["gt"] == "go straight" for it in items["e2"]) / n_e2 if n_e2 else None
    diag = []

    # ---------------- E1 ----------------
    e1, ci_rows = {}, []
    for m in models:
        for task in ("yesno", "action"):
            pool = pools[task]
            cell = {}
            for c in CONDITIONS:
                rs = rows(m, "e1", c, pool)
                sh, up = top_share(rs), unparsed_rate(rs, task)
                cell[c] = {"acc": acc(rs), "share": sh, "unp": up,
                           "flag_const": sh is not None and sh >= 0.95,
                           "flag_unp": up is not None and up > 0.05}
            vals = {it["uid"]: recs[(m, "e1", "original", it["uid"])]["correct"] -
                    recs[(m, "e1", "blank", it["uid"])]["correct"]
                    for it in pool if (m, "e1", "original", it["uid"]) in recs
                    and (m, "e1", "blank", it["uid"]) in recs}
            g, lo, hi = cluster_ci(vals, scene_of)
            artifact = any(cell[c][k] for c in ("original", "blank") for k in ("flag_const", "flag_unp"))
            e1[(m, task)] = {"cells": cell, "G": g, "lo": lo, "hi": hi, "artifact": artifact}
            ci_rows.append([MODELS[m][1], task, fmt(cell["original"]["acc"]), fmt(g), fmt(lo), fmt(hi),
                            "artifact" if artifact else "ok"])
            for c in CONDITIONS:
                if cell[c]["flag_const"]:
                    diag.append("E1 %s %s %s: >=95%% of responses are one class (%.1f%%)." %
                                (m, task, c, 100 * cell[c]["share"]))
                if cell[c]["flag_unp"]:
                    diag.append("E1 %s %s %s: %.1f%% of responses have no valid label." %
                                (m, task, c, 100 * cell[c]["unp"]))

    def cell_str(m, task, c):
        x = e1[(m, task)]["cells"][c]
        s = fmt(x["acc"])
        marks = ("$^{\\dagger}$" if x["flag_const"] else "") + ("$^{\\ddagger}$" if x["flag_unp"] else "")
        return s + marks

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{E1: Accuracy (\%) Under Four Input Conditions and Grounding Gap $G$ = Orig.\ $-$ Blank "
         r"(Percentage Points). Y/N = Yes/No Items ($n=" + str(n_yn) + r"$), Act. = Target-Action Items ($n=" +
         str(n_act) + r"$).}",
         r"\label{tab:cs_e1}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
         r"\begin{tabular}{@{}l l c c c c c@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Task} & \textbf{Orig.} & \textbf{Blank} & \textbf{Corrupt.} & "
         r"\textbf{No img.} & $G$ \\", r"\midrule",
         r"Majority answer & Y/N & " + " & ".join([fmt(maj)] * 4) + r" & 0.0 \\",
         r"Constant ``go straight'' & Act. & " + " & ".join([fmt(const)] * 4) + r" & 0.0 \\",
         r"\midrule"]
    for m in models:
        for j, task in enumerate(("yesno", "action")):
            x = e1[(m, task)]
            g = signed(x["G"])
            g = "(" + g + ")" if x["artifact"] else g
            L.append((MODELS[m][1] if j == 0 else "") + " & " + ("Y/N" if task == "yesno" else "Act.") + " & " +
                     " & ".join(cell_str(m, task, c) for c in CONDITIONS) + " & " + g + r" \\")
    L += [r"\bottomrule",
          r"\multicolumn{7}{@{}p{\columnwidth}@{}}{\scriptsize $^{\dagger}$At least 95\% of responses are the same "
          r"class. $^{\ddagger}$More than 5\% of responses contain no valid label and are scored incorrect. "
          r"A $G$ in parentheses involves such outputs and is not a grounding measure. 95\% scene-clustered "
          r"bootstrap CIs for $G$ are in the supplement.} \\",
          r"\end{tabular}", r"\end{table}"]
    table_e1 = "\n".join(L)

    # ---------------- E1b: behavior ----------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{E1: Prediction Behavior With Original Images. Yes = Share of Yes/No Items Answered ``Yes'' "
         r"(True Share: " + fmt(true_yes) + r"\%); Straight = Share of Target-Action Items Predicted ``Go "
         r"Straight'' (True Share: " + fmt(const) + r"\%); Bal.\ Acc.\ = Balanced Accuracy over the Four "
         r"Classes (Chance: 25.0\%).}",
         r"\label{tab:cs_e1b}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
         r"\begin{tabular}{@{}l c c c c c@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Yes (\%)} & \textbf{Yes, no img. (\%)} & \textbf{Straight (\%)} & "
         r"\textbf{Bal.\ Acc.\ (\%)} & \textbf{Unparsed (\%)} \\", r"\midrule"]
    behav = {}
    for m in models:
        yo = rows(m, "e1", "original", pools["yesno"])
        yn_ = rows(m, "e1", "noimage", pools["yesno"])
        ao = rows(m, "e1", "original", pools["action"])
        yes_o = 100.0 * np.mean([r["pred"] == "yes" for r in yo]) if yo else None
        yes_n = 100.0 * np.mean([r["pred"] == "yes" for r in yn_]) if yn_ else None
        straight = 100.0 * np.mean([r["pred"] == "go straight" for r in ao]) if ao else None
        ba = balanced_acc(ao)
        unp = 100.0 * (unparsed_rate(ao, "action") or 0)
        behav[m] = {"yes": yes_o, "yes_noimg": yes_n, "straight": straight, "bal_acc": ba, "unparsed": unp}
        L.append("%s & %s & %s & %s & %s & %s \\\\" % (MODELS[m][1], fmt(yes_o), fmt(yes_n), fmt(straight),
                                                       fmt(ba), fmt(unp)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_e1b = "\n".join(L)

    # ---------------- E2 ----------------
    e2, e2_csv, deltas = {}, [], {}
    e2_pool = items["e2"]
    for m in models:
        dr, rr = rows(m, "e2", "direct", e2_pool), rows(m, "e2", "reasoning", e2_pool)
        for mode, rs in (("direct", dr), ("reasoning", rr)):
            if not rs:
                e2[(m, mode)] = None
                continue
            ms = np.array([r["ms"] for r in rs])
            tk = np.array([r["tokens"] for r in rs], dtype=float)
            st = {"acc": acc(rs), "unp": 100.0 * unparsed_rate(rs, "action"),
                  "lenient": (100.0 * float(np.mean([lenient_pred(r) == r["gt"] for r in rs]))
                              if mode == "reasoning" else acc(rs)),
                  "trunc": 100.0 * float(np.mean([is_truncated(r) for r in rs])),
                  "tok": float(tk.mean()), "p50": float(np.percentile(ms, 50)),
                  "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
                  "max": float(ms.max()), "slope": None, "intercept": None, "r2": None,
                  "n_in": float(np.mean([r.get("n_input_tokens", np.nan) for r in rs]))}
            if mode == "reasoning" and len(set(tk)) > 1:
                b, a = np.polyfit(tk, ms, 1)
                st.update(slope=float(b), intercept=float(a), r2=float(np.corrcoef(tk, ms)[0, 1] ** 2))
            e2[(m, mode)] = st
            if st["trunc"] > 5:
                diag.append("E2 %s %s: %.1f%% of responses hit the token cap." % (m, mode, st["trunc"]))
            if st["unp"] > 5:
                diag.append("E2 %s %s: %.1f%% of responses have no valid label." % (m, mode, st["unp"]))
            e2_csv.append([MODELS[m][1], mode, fmt(st["acc"]), fmt(st["lenient"]), fmt(st["unp"]), fmt(st["trunc"]),
                           fmt(st["tok"]), fmt(st["p50"], 0), fmt(st["p95"], 0), fmt(st["p99"], 0),
                           fmt(st["max"], 0), fmt(st["slope"], 2), fmt(st["intercept"], 0), fmt(st["r2"], 3)])
        pair = {it["uid"]: recs[(m, "e2", "reasoning", it["uid"])]["correct"] -
                recs[(m, "e2", "direct", it["uid"])]["correct"]
                for it in e2_pool if (m, "e2", "reasoning", it["uid"]) in recs
                and (m, "e2", "direct", it["uid"]) in recs}
        deltas[m] = cluster_ci(pair, scene_of) if pair else (None, None, None)

    caps = sorted({r.get("max_new_tokens") for k, r in recs.items() if k[1] == "e2" and k[2] == "reasoning"} - {None})
    caps_d = sorted({r.get("max_new_tokens") for k, r in recs.items() if k[2] == "direct"} - {None})
    cap_txt = str(caps[0]) if len(caps) == 1 else ("%d" % LEGACY["reason"] if not caps else "/".join(map(str, caps)))

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{E2: Accuracy, Generated Tokens, and End-to-End Latency on Target-Action Items ($n=" + str(n_e2) +
         r"$). Constant ``Go Straight'' Baseline: " + fmt(const_e2) + r"\% on This Subset. Responses Without a "
         r"Valid Label Are Scored Incorrect.}",
         r"\label{tab:cs_e2}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
         r"\begin{tabular}{@{}l l c c c c c c@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Prompt} & \textbf{Acc.\ (\%)} & \textbf{Unp.\ (\%)} & \textbf{Tokens} & "
         r"\textbf{p50 (ms)} & \textbf{p99 (ms)} & \textbf{ms/tok} \\", r"\midrule"]
    for m in models:
        for j, mode in enumerate(("direct", "reasoning")):
            s = e2.get((m, mode))
            label = MODELS[m][1] if j == 0 else ""
            ml = "Direct" if mode == "direct" else "Reasoning"
            if s is None:
                L.append(label + " & " + ml + r" & -- & -- & -- & -- & -- & -- \\")
            else:
                L.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                    label, ml, fmt(s["acc"]) + ("$^{\\dagger}$" if s["unp"] > 5 else ""), fmt(s["unp"]), fmt(s["tok"]), fmt(s["p50"], 0),
                    fmt(s["p99"], 0), fmt(s["slope"]) if s["slope"] is not None else "--"))
    L += [r"\bottomrule",
          r"\multicolumn{8}{@{}p{\columnwidth}@{}}{\scriptsize ms/tok: slope of per-item latency on generated "
          r"tokens for the reasoning prompt (intercept = prefill and preprocessing). $^{\\dagger}$More than 5\\% of "
          r"responses contain no valid final answer (truncated or format not followed) and are scored incorrect; "
          r"this is a format failure, not a reasoning result. Reasoning outputs are capped "
          r"at " + cap_txt + r" new tokens. p99 over $n=" + str(n_e2) + r"$ is set by the three slowest runs.} \\",
          r"\end{tabular}", r"\end{table}"]
    table_e2 = "\n".join(L)

    # ---------------- Text ----------------
    def uniq(key):
        return sorted(set(v.get(key, "") for v in meta.values() if isinstance(v, dict)) - {""})
    gpu = ", ".join(uniq("gpu")) or "[GPU]"
    setup = (
        r"\textit{Data.} Items come from the DriveLM-nuScenes v1.1 training split~\cite{sima2024drivelm}. "
        r"We evaluate " + str(n_yn) + r" yes/no questions and " + str(n_act) + r" target-action questions, "
        r"each drawn at random with seed " + str(items.get("seed", 0)) + r" from items whose six camera images "
        r"are available. Target-action answers are mapped to four classes (go straight, stationary, turn left, "
        r"and turn right), and items in other classes are excluded. The six views are resized to "
        r"$448\times252$ pixels and tiled into one $2\times3$ image; each model's processor then resizes this "
        r"image to its native input resolution." + "\n\n" +
        r"\textit{Inference.} Decoding is greedy, weights are in bf16, and the batch size is 1, on a single " + gpu +
        r" GPU with PyTorch " + ", ".join(uniq("torch")) + r" and Transformers " + ", ".join(uniq("transformers")) +
        r". Direct answers are limited to " + str(LEGACY["direct"] if not caps_d else "/".join(map(str, caps_d))) + r" new tokens and reasoning outputs to " +
        cap_txt + r". A response without a valid label is scored incorrect. Latency is measured end to end "
        r"(image preprocessing, prefill, and decoding, excluding tiling) after warm-up. Prompts are fixed across "
        r"conditions and listed in the supplementary material."
    )
    e2_sentence = (r"A random subset of " + str(n_e2) + r" target-action items is answered with two prompts, "
                   r"a direct answer and ``reason step by step, then answer.''")

    ok_g = [e1[(m, "yesno")]["G"] for m in models]
    noimg_yn = [e1[(m, "yesno")]["cells"]["noimage"]["acc"] for m in models]
    orig_act = {m: e1[(m, "action")]["cells"]["original"]["acc"] for m in models}
    above = [MODELS[m][1] for m in models if orig_act[m] > const + 1e-9]
    collapsed = [MODELS[m][1] for m in models
                 if e1[(m, "action")]["cells"]["blank"]["flag_const"] or e1[(m, "action")]["cells"]["blank"]["flag_unp"]
                 or e1[(m, "action")]["cells"]["original"]["flag_const"]]
    slopes = [e2[(m, "reasoning")]["slope"] for m in models if e2.get((m, "reasoning")) and e2[(m, "reasoning")]["slope"]]
    r2s = [e2[(m, "reasoning")]["r2"] for m in models if e2.get((m, "reasoning")) and e2[(m, "reasoning")]["r2"]]
    tok_ratio, p50_ratio = [], []
    for m in models:
        a, b = e2.get((m, "direct")), e2.get((m, "reasoning"))
        if a and b and a["tok"] > 0 and a["p50"] > 0:
            tok_ratio.append(b["tok"] / a["tok"])
            p50_ratio.append(b["p50"] / a["p50"])
    dparts, fmt_fail = [], []
    for m in models:
        g, lo, hi = deltas[m]
        st = e2.get((m, "reasoning"))
        if g is None:
            continue
        if st and (st["unp"] > 5):
            fmt_fail.append("%s (%.0f\\%% unparsed)" % (MODELS[m][1], st["unp"]))
        else:
            dparts.append("%s %s (95\\%% CI %s to %s)" % (MODELS[m][1], signed(g), signed(lo), signed(hi)))
    trunc_models = [MODELS[m][1] for m in models if e2.get((m, "reasoning")) and
                    (e2[(m, "reasoning")]["trunc"] > 5 or e2[(m, "reasoning")]["unp"] > 5)]

    results = (
        "% DRAFT generated from your results. Check every claim before use.\n"
        + (("%% WARNING: reasoning outputs of %s are truncated or unparsed; re-run E2 with a larger cap before using "
            "its accuracy change.\n" % ", ".join(trunc_models)) if trunc_models else "")
        + r"Tables~\ref{tab:cs_e1}, \ref{tab:cs_e1b}, and~\ref{tab:cs_e2} show the behavior of the " + str(len(models)) +
        r" models. On yes/no items, replacing the images with a blank frame lowers accuracy by " + span(ok_g) +
        r" points (some of it reflects answer priors, see Table~\ref{tab:cs_e1b}), yet accuracy without any image remains " + span(noimg_yn) +
        r"\%, against a " + fmt(maj) + r"\% majority-answer baseline. On target-action items, " +
        ("no model exceeds" if not above else "only " + ", ".join(above) + " exceeds") +
        r" the " + fmt(const) + r"\% constant ``go straight'' baseline (" + ", ".join(fmt(orig_act[m]) for m in models) +
        r"\%)" + ((r"; the action responses of " + ", ".join(collapsed) + r" are near-constant (at least 95\% one class) or "
                   r"often unparsed under some conditions, so their action grounding gaps are not interpreted") if collapsed else "") +
        r". Latency grows linearly with generated tokens at " + span(slopes) + r"~ms per token ($R^2\ge" +
        fmt(min(r2s) if r2s else None, 3) + r"$), so step-by-step reasoning multiplies median latency by " +
        span(p50_ratio) + r" and generated tokens by " + span(tok_ratio) + r". Reasoning changed target-action "
        r"accuracy relative to direct answering by " + "; ".join(dparts) + r" points, against a " + fmt(const_e2) +
        r"\% constant baseline on this subset" +
        ((r"; for " + ", ".join(fmt_fail) + r" the final answer was missing too often to compare accuracies") if fmt_fail else "") +
        r". "
        r"[Add one or two sentences on what this means for Sections~\ref{ssec:lang_eval} and~\ref{ssec:implications}.]"
    )
    abstract = (
        "% DRAFT abstract sentence. Replace the case-study sentence in the abstract.\n"
        r"A controlled case study of " + str(len(models)) + r" open VLMs on DriveLM-nuScenes shows that " +
        ("none exceeds" if not above else "only " + ", ".join(above) + " exceeds") +
        r" a constant ``go straight'' baseline on target-action items, that yes/no accuracy without any image "
        r"remains " + span(noimg_yn) + r"\%, and that step-by-step reasoning multiplies median latency by " +
        span(p50_ratio) + r" (about " + fmt(float(np.mean(slopes)) if slopes else None, 0) +
        r"~ms per generated token)."
    )

    out = ["% ===== 1. Replace the Data and Inference paragraphs of Section 7.1 with: =====", setup,
           "\n% ===== 2. Replace the first sentence of Section 7.3 (E2) with: =====", e2_sentence,
           "\n% ===== 3. Replace the table environment tab:cs_e1 with: =====", table_e1,
           "\n% ===== 3b. Add after tab:cs_e1: =====", table_e1b,
           "\n% ===== 4. Replace the table environment tab:cs_e2 with: =====", table_e2,
           "\n% ===== 5. Replace the red paragraph in Section 7.4 with (after editing): =====", results,
           "\n% ===== 6. Abstract sentence (after editing): =====", abstract]
    text = "\n".join(out)
    with open(os.path.join(d, "section7_snippets.tex"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(d, "ci.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "acc_original", "G", "G_ci_low", "G_ci_high", "status"])
        w.writerows(ci_rows)
    with open(os.path.join(d, "e2_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt", "acc_strict", "acc_lenient_lastline", "unparsed_pct", "truncated_pct", "tokens_mean", "p50_ms", "p95_ms",
                    "p99_ms", "max_ms", "ms_per_token", "intercept_ms", "r2"])
        w.writerows(e2_csv)
        w.writerow([])
        w.writerow(["model", "reasoning_minus_direct_pp", "ci_low", "ci_high"])
        for m in models:
            g, lo, hi = deltas[m]
            w.writerow([MODELS[m][1], fmt(g), fmt(lo), fmt(hi)])
    with open(os.path.join(d, "diagnostics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(diag) + "\n" if diag else "No flags.\n")
    print(text)
    print("\n=== DIAGNOSTICS (read before using any number) ===")
    print("\n".join(diag) if diag else "No flags.")
    print("\nWrote section7_snippets.tex, ci.csv, e2_stats.csv, diagnostics.txt in %s" % d)


# ---------------------------------------------------------------------------
# System list summary (Section 3)
# ---------------------------------------------------------------------------

def cmd_summary(args):
    n = len(SYSTEMS)
    print("Systems: %d" % n)
    print("Coupling :", dict(Counter(s[2] for s in SYSTEMS)))
    print("Reasoning:", dict(Counter(s[3] for s in SYSTEMS)))
    grid = Counter((s[2], s[3]) for s in SYSTEMS)
    print("\nCoupling x reasoning grid (count):")
    print("      " + " ".join("%4s" % r for r in REASONING_NAMES))
    for c in COUPLING_NAMES:
        print("  %s   " % c + " ".join("%4d" % grid[(c, r)] for r in REASONING_NAMES))

    def year(v):
        m = re.search(r"(20\d\d)", v)
        return int(m.group(1)) if m else None
    print("\nSystems per year:", dict(sorted(Counter(year(s[1]) for s in SYSTEMS).items())))
    if RL_POSTTRAINED:
        names = {s[0]: s for s in SYSTEMS}
        missing = [x for x in RL_POSTTRAINED if x not in names]
        if missing:
            print("Unknown names in RL_POSTTRAINED: %s" % missing)
        yrs = sorted((year(names[x][1]), x) for x in RL_POSTTRAINED if x in names)
        print("\nRL post-training: %d systems" % len(yrs))
        for y, x in yrs:
            print("   %s  %s" % (y, x))
        print("All in 2025 or 2026: %s" % all(y is not None and y >= 2025 for y, _ in yrs))
    else:
        print("\nRL_POSTTRAINED is empty: fill it from the primary publications, then re-run, "
              "before quoting a number of RL post-trained systems.")


# ---------------------------------------------------------------------------
# Inter-rater agreement (Section 3.4 and screening)
# ---------------------------------------------------------------------------

def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None, None
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    k = 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)
    return k, po


def kappa_ci(a, b, B=2000, seed=0):
    n = len(a)
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        k, _ = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        ks.append(k)
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def kappa_str(k, ci):
    return "n/a" if k is None else "%.2f (95%% CI %.2f to %.2f)" % (k, ci[0], ci[1])


def norm_coupling(v):
    v = (v or "").strip().upper()
    for k, name in COUPLING_NAMES.items():
        if v == k or v == name.upper():
            return k
    return v


def norm_reasoning(v):
    v = (v or "").strip().upper().replace(" ", "")
    if re.fullmatch(r"[0-4]", v):
        return "R" + v
    return v


def cmd_kappa_template(args):
    os.makedirs(args.outdir, exist_ok=True)
    for name, seed in (("annotator_A.csv", 1), ("annotator_B.csv", 2)):
        rows = [(s, v) for s, v, _, _, _ in SYSTEMS]
        random.Random(seed).shuffle(rows)
        with open(os.path.join(args.outdir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["system", "venue", "coupling", "reasoning", "notes"])
            for s, v in rows:
                w.writerow([s, v, "", "", ""])
    with open(os.path.join(args.outdir, "definitions.txt"), "w", encoding="utf-8") as f:
        f.write(DEFINITIONS)
    print("Wrote annotator_A.csv, annotator_B.csv, and definitions.txt to %s." % args.outdir)
    print("Each annotator fills 'coupling' (I, M, D, U) and 'reasoning' (R0 to R4) independently,")
    print("from the primary publications, WITHOUT seeing Table IV. Keep these files unchanged afterwards.")


def read_annotations(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["system"].strip()] = (norm_coupling(r.get("coupling")), norm_reasoning(r.get("reasoning")))
    return out


def write_confusion(path, a, b, labels):
    cnt = Counter(zip(a, b))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A \\ B"] + labels)
        for x in labels:
            w.writerow([x] + [cnt[(x, y)] for y in labels])


def cmd_kappa(args):
    if np is None:
        sys.exit("kappa needs numpy.")
    A, B = read_annotations(args.a), read_annotations(args.b)
    names = [s for s, _, _, _, _ in SYSTEMS if s in A and s in B]
    missing = [s for s, _, _, _, _ in SYSTEMS if s not in names]
    if missing:
        print("Warning: missing in one sheet: %s" % ", ".join(missing))
    bad = [(s, A[s], B[s]) for s in names
           if A[s][0] not in COUPLING_NAMES or B[s][0] not in COUPLING_NAMES
           or A[s][1] not in REASONING_NAMES or B[s][1] not in REASONING_NAMES]
    if bad:
        print("Invalid or empty labels (fix before reporting):")
        for s, a, b in bad:
            print("   %s  A=%s  B=%s" % (s, a, b))
        sys.exit("Aborting: kappa is only meaningful with complete, valid labels.")
    if not names:
        sys.exit("No systems in common between the two sheets.")
    ca, cb = [A[s][0] for s in names], [B[s][0] for s in names]
    ra, rb = [A[s][1] for s in names], [B[s][1] for s in names]
    kc, pc = cohen_kappa(ca, cb)
    kr, pr = cohen_kappa(ra, rb)
    kj, pj = cohen_kappa([x + y for x, y in zip(ca, ra)], [x + y for x, y in zip(cb, rb)])
    cic, cir = kappa_ci(ca, cb), kappa_ci(ra, rb)
    print("Systems compared: %d" % len(names))
    print("Coupling : kappa = %s, raw agreement = %.1f%%" % (kappa_str(kc, cic), 100 * pc))
    print("Reasoning: kappa = %s, raw agreement = %.1f%%" % (kappa_str(kr, cir), 100 * pr))
    print("Joint    : kappa = %.3f, raw agreement = %.1f%%" % (kj, 100 * pj))
    print("(With few items and one dominant class, kappa can be low although raw agreement is high; "
          "report both.)")
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        write_confusion(os.path.join(args.outdir, "confusion_coupling.csv"), ca, cb, list(COUPLING_NAMES))
        write_confusion(os.path.join(args.outdir, "confusion_reasoning.csv"), ra, rb, list(REASONING_NAMES))
        print("Wrote confusion matrices to %s (put them in the supplement)." % args.outdir)
    print("\nDisagreements to resolve by discussion:")
    for s in names:
        if A[s] != B[s]:
            print("   %-20s A=%s/%s  B=%s/%s" % (s, A[s][0], A[s][1], B[s][0], B[s][1]))
    ref = {s: (c, r) for s, _, c, r, _ in SYSTEMS}
    for tag, X in (("A", A), ("B", B)):
        agree = sum(X[s] == ref[s] for s in names)
        print("Annotator %s matches Table IV on %d/%d systems." % (tag, agree, len(names)))
    sentence = ("Two authors classified all systems independently from the primary publications. Before "
                "resolution, Cohen's $\\kappa$ was %.2f (95\\%% CI %.2f to %.2f) for coupling and %.2f "
                "(95\\%% CI %.2f to %.2f) for reasoning (raw agreement %.0f\\%% and %.0f\\%%, $n=%d$), and "
                "disagreements were resolved by discussion."
                % (kc, cic[0], cic[1], kr, cir[0], cir[1], 100 * pc, 100 * pr, len(names)))
    print("\nPaste into Section 3.4 (replaces the red marker and the sentence before it):\n")
    print(sentence)


def cmd_screen_kappa(args):
    """Agreement of two reviewers on title/abstract screening (binary include/exclude)."""
    if np is None:
        sys.exit("screen-kappa needs numpy.")

    def read(path):
        out = {}
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = (r.get(args.column) or "").strip().lower()
                if v:
                    out[r[args.key].strip()] = "include" if _yes(v) else "exclude"
        return out
    A, B = read(args.a), read(args.b)
    ids = sorted(set(A) & set(B))
    if not ids:
        sys.exit("No screened records in common.")
    a, b = [A[i] for i in ids], [B[i] for i in ids]
    k, po = cohen_kappa(a, b)
    ci = kappa_ci(a, b)
    print("Records compared: %d" % len(ids))
    print("kappa = %s, raw agreement = %.1f%%" % (kappa_str(k, ci), 100 * po))
    print("\nPaste (fill the stage name):\n")
    print("Inter-reviewer agreement at title/abstract screening before resolution was $\\kappa=%.2f$ "
          "(95\\%% CI %.2f to %.2f; raw agreement %.0f\\%%, $n=%d$ screened records)."
          % (k, ci[0], ci[1], 100 * po, len(ids)))


# ---------------------------------------------------------------------------
# PRISMA search and counting (Section 1.2)
# ---------------------------------------------------------------------------

MODEL_TERMS = ['"vision-language model"', 'VLM', '"vision-language-action"', 'VLA',
               '"large language model"', 'LLM', '"multimodal LLM"']
DOMAIN_TERMS = ['"autonomous driving"', '"self-driving"', '"end-to-end driving"', '"motion planning"']
ARXIV_CATS = ["cs.CV", "cs.RO", "cs.AI", "cs.CL"]
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}


def arxiv_page(query, start, size, email, id_list=None):
    import urllib.parse
    import urllib.request
    if id_list:
        params = urllib.parse.urlencode({"id_list": ",".join(id_list), "max_results": len(id_list)})
    else:
        params = urllib.parse.urlencode({"search_query": query, "start": start, "max_results": size,
                                         "sortBy": "submittedDate", "sortOrder": "ascending"})
    req = urllib.request.Request(ARXIV_API + "?" + params,
                                 headers={"User-Agent": "survey-tools/2.0 (%s)" % email})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def cmd_prisma_search(args):
    import datetime
    import xml.etree.ElementTree as ET
    start_d = args.start.replace("-", "") + "0000"
    end_d = args.end.replace("-", "") + "2359"
    cats = " OR ".join("cat:%s" % c for c in ARXIV_CATS)
    records, log = {}, []
    for mt in MODEL_TERMS:
        for dt in DOMAIN_TERMS:
            q = "all:%s AND all:%s AND submittedDate:[%s TO %s] AND (%s)" % (mt, dt, start_d, end_d, cats)
            start, total, tries = 0, None, 0
            while total is None or start < total:
                try:
                    root = ET.fromstring(arxiv_page(q, start, args.page_size, args.email))
                except Exception as e:  # network hiccup
                    tries += 1
                    if tries > 3:
                        print("  giving up on page %d of %s %s: %s" % (start, mt, dt, e))
                        break
                    time.sleep(10)
                    continue
                total = int(root.find("os:totalResults", NS).text)
                entries = root.findall("a:entry", NS)
                if not entries and start < total:
                    tries += 1
                    if tries > 3:
                        break
                    time.sleep(10)
                    continue
                tries = 0
                for e in entries:
                    aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
                    rec = records.setdefault(aid, {
                        "source": "arXiv", "id": aid,
                        "title": " ".join(e.find("a:title", NS).text.split()),
                        "year": e.find("a:published", NS).text[:4],
                        "categories": " ".join(c.get("term") for c in e.findall("a:category", NS)),
                        "abstract": " ".join(e.find("a:summary", NS).text.split()),
                        "matched_queries": "", "screen_title_abstract": "",
                        "fulltext_assessed": "", "included": "", "exclusion_reason": ""})
                    rec["matched_queries"] = (rec["matched_queries"] + "; " if rec["matched_queries"] else "") + "%s + %s" % (mt, dt)
                start += len(entries)
                time.sleep(3.1)  # arXiv API etiquette
            log.append({"model_term": mt, "domain_term": dt, "query": q, "total_results": total})
            print("  %-28s + %-22s -> %s results" % (mt, dt, total))
    cols = ["source", "id", "title", "year", "categories", "abstract", "matched_queries",
            "screen_title_abstract", "fulltext_assessed", "included", "exclusion_reason"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(records.values(), key=lambda x: x["id"]):
            w.writerow(r)
    log_path = os.path.splitext(args.out)[0] + "_search_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"search_date": datetime.date.today().isoformat(), "api": ARXIV_API,
                   "date_range": [args.start, args.end], "categories": ARXIV_CATS,
                   "unique_records": len(records), "queries": log}, f, indent=1)
    print("\nWrote %d unique arXiv records to %s and the search log to %s." % (len(records), args.out, log_path))
    print("Put the search date and the query strings (from the log) in the supplement. Screen the records by "
          "filling 'screen_title_abstract' (include/exclude); for included rows fill 'included' (yes/no) and "
          "'exclusion_reason'. Add records from CVF, OpenReview, IEEE Xplore, and ACM DL as extra CSVs with "
          "the same columns, or state that the automated search covers arXiv only.")


def cmd_check_links(args):
    import xml.etree.ElementTree as ET
    ids = [(s, a) for s, _, _, _, a in SYSTEMS if a]
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        try:
            root = ET.fromstring(arxiv_page(None, 0, 0, args.email, id_list=[a for _, a in chunk]))
        except Exception as e:
            print("Request failed: %s" % e)
            continue
        found = {}
        for e in root.findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
            found[aid] = " ".join(e.find("a:title", NS).text.split())
        for s, a in chunk:
            print("%-18s %-12s %s" % (s, a, found.get(a, "NOT FOUND")))
        time.sleep(3.1)
    print("\nCompare every title with the system name by eye. A mismatch means a wrong id.")


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _yes(v):
    return str(v).strip().lower() in ("include", "included", "yes", "y", "1", "true")


def _no(v):
    return str(v).strip().lower() in ("exclude", "excluded", "no", "n", "0", "false")


def cmd_prisma_count(args):
    rows = []
    for p in args.sheets:
        with open(p, "r", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    identified = len(rows)
    by_source = Counter(r.get("source", "unknown") or "unknown" for r in rows)
    seen, dedup = set(), []
    for r in rows:
        k = norm_title(r.get("title"))
        if k and k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    dups = identified - len(dedup)
    screened = len(dedup)
    unscreened = sum(1 for r in dedup if not _yes(r.get("screen_title_abstract")) and not _no(r.get("screen_title_abstract")))
    excl_screen = sum(1 for r in dedup if _no(r.get("screen_title_abstract")))
    full = [r for r in dedup if _yes(r.get("screen_title_abstract"))]
    included = sum(1 for r in full if _yes(r.get("included")))
    excl_full = sum(1 for r in full if _no(r.get("included")))
    undecided = len(full) - included - excl_full
    reasons = Counter((r.get("exclusion_reason") or "unspecified").strip() for r in full if _no(r.get("included")))
    if unscreened or undecided:
        print("Warning: %d records not screened and %d full texts undecided." % (unscreened, undecided))
    print("Identified: %d (%s)" % (identified, ", ".join("%s %d" % kv for kv in by_source.items())))
    print("Duplicates removed: %d" % dups)
    print("Screened (title/abstract): %d, excluded: %d" % (screened, excl_screen))
    print("Full texts assessed: %d, excluded: %d" % (len(full), excl_full))
    print("Included: %d" % included)
    sentence = ("The search returned %d records, of which %d were duplicates. We screened %d titles "
                "and abstracts, assessed %d full texts, and included %d works, of which %d are "
                "classified in Table~\\ref{tab:master} (supplementary Table~S1)."
                % (identified, dups, screened, len(full), included, len(SYSTEMS)))
    reason_lines = " \\\\\n".join("\\quad %s & %d" % (k.replace("&", "\\&"), v) for k, v in reasons.most_common())
    table = "\n".join([
        r"\begin{table}[h]", r"\centering",
        r"\caption{Literature Selection (PRISMA-Style Flow)}", r"\label{tab:s1}",
        r"\begin{tabular}{@{}l r@{}}", r"\toprule",
        r"\textbf{Stage} & \textbf{Records} \\", r"\midrule",
        "\n".join(r"Identified via %s & %d \\" % (s, n) for s, n in by_source.items()),
        r"Duplicates removed & %d \\" % dups,
        r"Titles and abstracts screened & %d \\" % screened,
        r"Excluded at screening & %d \\" % excl_screen,
        r"Full texts assessed & %d \\" % len(full),
        r"Excluded at full-text stage & %d \\" % excl_full,
        (reason_lines + r" \\") if reason_lines else "",
        r"\midrule",
        r"Included & %d \\" % included,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    print("\nPaste into Section 1.2 (replaces the red marker):\n")
    print(sentence)
    print("\nSupplementary Table S1:\n")
    print(table)


# ---------------------------------------------------------------------------
# README for the companion repository (Conclusion)
# ---------------------------------------------------------------------------

def cmd_readme(args):
    lines = ["# VLMs and VLAs for Autonomous Driving: Paper List", "",
             "Companion list for the survey *From Seeing to Acting: A Survey of Vision-Language "
             "and Vision-Language-Action Models for Autonomous Driving*. Systems are classified "
             "along two axes: language-action coupling (I, M, D, U) and reasoning regime (R0 to R4).",
             "", "## Taxonomy", ""]
    for k, v in COUPLING_NAMES.items():
        lines.append("- **%s**: %s" % (k, v))
    lines.append("")
    for k, v in REASONING_NAMES.items():
        lines.append("- **%s**: %s" % (k, v))
    for cls in ("I", "M", "D", "U"):
        lines += ["", "## %s (%s)" % (COUPLING_NAMES[cls], cls), "",
                  "| System | Venue | Reasoning | Paper |", "|---|---|---|---|"]
        for s, v, c, r, ax in SYSTEMS:
            if c != cls:
                continue
            link = "[arXiv](https://arxiv.org/abs/%s)" % ax if ax else ""
            lines.append("| %s | %s | %s | %s |" % (s, v, r, link))
    lines += ["", "## Case-study code", "",
              "`survey_tools.py` reproduces the Section 7 case study (prompts in `prompts.json`, item ids in "
              "`items.json`, raw responses in `results.jsonl`). DriveLM and nuScenes data are not redistributed; "
              "obtain them from the original sources and respect their licenses.",
              "", "## Citation", "", "```bibtex", "@article{attri2026seeing,",
              "  title   = {From Seeing to Acting: A Survey of Vision-Language and Vision-Language-Action Models for Autonomous Driving},",
              "  author  = {Attri, Anuj and Verma, Anubhav and Thandavarayan, Gokulnath},",
              "  journal = {Under review},", "  year    = {2026}", "}", "```", "",
              "Pull requests that add papers are welcome. Please give the coupling class and reasoning regime."]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote %s. Verify every link (check-links), push it to a public GitHub repository, and put the "
          "repository URL in the Conclusion." % args.out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Reproducibility tools for the VLM/VLA driving survey.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("drivelm-stats", help="recompute the DriveLM target-action distribution")
    s.add_argument("--drivelm-json", required=True)
    s.add_argument("--top", type=int, default=8)
    s.set_defaults(func=cmd_drivelm_stats)

    s = sub.add_parser("casestudy", help="run experiments E1 and E2")
    s.add_argument("--drivelm-json", required=True)
    s.add_argument("--image-root", action="append", required=True,
                   help="nuScenes root containing samples/CAM_*/ (repeatable)")
    s.add_argument("--outdir", required=True)
    s.add_argument("--models", default=",".join(MODELS))
    s.add_argument("--experiments", default="e1,e2")
    s.add_argument("--n-yesno", type=int, default=1000)
    s.add_argument("--n-action", type=int, default=1000)
    s.add_argument("--n-e2", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-new-tokens-direct", type=int, default=DEFAULT_MAX_DIRECT)
    s.add_argument("--max-new-tokens-reason", type=int, default=DEFAULT_MAX_REASON)
    s.add_argument("--attn", default="sdpa", help="attention implementation (sdpa, flash_attention_2, eager)")
    s.add_argument("--llava-pad-square", action="store_true",
                   help="pad the tiled image to a square for LLaVA-1.5 (avoids its 336x336 center crop); "
                        "sensitivity check only")
    s.set_defaults(func=cmd_casestudy)

    s = sub.add_parser("merge", help="replace blocks of a run by a re-run")
    s.add_argument("--base", required=True)
    s.add_argument("--rerun", required=True)
    s.add_argument("--outdir", required=True)
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("make-tables", help="build Section 7 tables and text")
    s.add_argument("--outdir", required=True)
    s.add_argument("--legacy-cap-direct", type=int, default=0,
                   help="token cap of the run for records without a truncated flag (e.g. 64)")
    s.add_argument("--legacy-cap-reason", type=int, default=0,
                   help="same, for reasoning outputs (e.g. 512)")
    s.set_defaults(func=cmd_make_tables)

    s = sub.add_parser("summary", help="count systems by class and year")
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("kappa-template", help="write annotation sheets")
    s.add_argument("--outdir", default="kappa")
    s.set_defaults(func=cmd_kappa_template)

    s = sub.add_parser("kappa", help="compute Cohen's kappa for the classification")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--outdir", default="kappa_out")
    s.set_defaults(func=cmd_kappa)

    s = sub.add_parser("screen-kappa", help="Cohen's kappa for two screening sheets")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--key", default="id")
    s.add_argument("--column", default="screen_title_abstract")
    s.set_defaults(func=cmd_screen_kappa)

    s = sub.add_parser("prisma-search", help="query arXiv and write a screening sheet")
    s.add_argument("--out", default="screening_arxiv.csv")
    s.add_argument("--start", default="2022-01-01")
    s.add_argument("--end", default="2026-06-30")
    s.add_argument("--page-size", type=int, default=200)
    s.add_argument("--email", default="contact@example.org", help="contact e-mail for the User-Agent")
    s.set_defaults(func=cmd_prisma_search)

    s = sub.add_parser("prisma-count", help="count PRISMA numbers from screened sheets")
    s.add_argument("sheets", nargs="+")
    s.set_defaults(func=cmd_prisma_count)

    s = sub.add_parser("check-links", help="print the arXiv title for each arXiv id")
    s.add_argument("--email", default="contact@example.org")
    s.set_defaults(func=cmd_check_links)

    s = sub.add_parser("readme", help="write README.md for the paper-list repository")
    s.add_argument("--out", default="README.md")
    s.set_defaults(func=cmd_readme)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
