import os
import json
import csv
import base64
import requests
import re
import numpy as np
import cv2
import matplotlib.image as mpimg
from scipy.spatial.distance import cdist
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from skimage.feature import local_binary_pattern
from skimage import io as sio, color
from shapely.geometry import Polygon, Point
from legendParser.tool_pool.map_legend_detector import map_legend_detector
from legendParser.tool_pool.map_component_detector import map_component_detector

# =====================================================
# Parameter Configuration
# =====================================================
DOWNSCALE_TARGET = 2500
DEFAULT_SCALE = 1400
DEFAULT_SIGMA = 0.8
DEFAULT_MIN_SIZE = 400

SAVE_DIR = "./output"
LEGEND_ITEM_DIR = os.path.join(SAVE_DIR, "legend_items")
WHITE_THRESH = 245

GEMINI_API_URL = " "
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
FALLBACK_GEMINI_KEY = " "

GEMINI_HEADERS = {
    "Authorization": GEMINI_API_KEY if GEMINI_API_KEY else (FALLBACK_GEMINI_KEY or ""),
    "Content-Type": "application/json"
}

# =====================================================
# Basic Utilities
# =====================================================
def safe_parse_json(text):
    if not text: return None
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


def _is_null_like(x):
    if x is None: return True
    if isinstance(x, str) and x.strip().lower() == "null": return True
    return False


def is_null_geology_symbol(sym):
    import re as _re
    def _has_tofu_or_pua(s: str) -> bool:
        if not s: return False
        if "□" in s: return True
        return _re.search(r"[\uE000-\uF8FF]", s) is not None

    if _is_null_like(sym): return True
    if isinstance(sym, str):
        s = sym.strip()
        s_low = s.lower()
        if _has_tofu_or_pua(s): return True
        if s_low in {"unknown", "unrecognized", "n/a", "na"}: return True
        if s in {"未识别到地质符号", "未识别", "识别失败", "无法识别"}: return True
        if "未识别到地质符号" in s: return True
        if "unknown" in s_low: return True
        if "" in s: return True
        return False

    if isinstance(sym, dict):
        for _, v in sym.items():
            if isinstance(v, str) and _has_tofu_or_pua(v.strip()): return True
        cand = []
        for k in ["final_symbol", "terminal", "base", "superscript", "subscript", "final"]:
            v = sym.get(k)
            if isinstance(v, str):
                cand.append(v.strip())
            else:
                cand.append(v)
        if all(_is_null_like(v) or (isinstance(v, str) and v.strip() == "") for v in cand):
            return True
        for k in ["final_symbol", "terminal", "base", "final"]:
            v = sym.get(k)
            if isinstance(v, str):
                s = v.strip()
                s_low = s.lower()
                if _has_tofu_or_pua(s): return True
                if s_low in {"unknown", "unrecognized", "n/a", "na"}: return True
                if s in {"未识别到地质符号", "未识别", "识别失败", "无法识别"}: return True
                if "未识别到地质符号" in s: return True
                if "unknown" in s_low: return True
                if "" in s: return True
        for k in ["superscript", "subscript"]:
            v = sym.get(k)
            if isinstance(v, str):
                s = v.strip()
                if _has_tofu_or_pua(s) or ("" in s): return True
        return False
    return False


def _ensure_text_fallback(txt):
    if isinstance(txt, dict): return txt
    return {"legend_text": "", "confidence": 0.0}


def _ensure_symbol_fallback(sym):
    if isinstance(sym, dict): return sym
    return {"final_symbol": "Unknown", "confidence": 0.0, "base": None, "superscript": None, "subscript": None}


def image_to_base64(image_array_rgb):
    img_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def gemini_call(image_crop_rgb, prompt):
    if not GEMINI_HEADERS["Authorization"]: return ""
    try:
        b64 = image_to_base64(image_crop_rgb)
        payload = {
            "model": GEMINI_MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url",
                                                                                         "image_url": {
                                                                                             "url": f"data:image/jpeg;base64,{b64}"}}]}]
        }
        resp = requests.post(GEMINI_API_URL, headers=GEMINI_HEADERS, json=payload, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] Gemini error: {e}")
        return ""


def bbox_to_mask(shape, bbox, pad=0):
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0 - pad));
    y0 = max(0, int(y0 - pad))
    x1 = min(w, int(x1 + pad));
    y1 = min(h, int(y1 + pad))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def get_component_bbox(comp_dict, key, choose="largest"):
    boxes = comp_dict.get(key, [])
    if not boxes: return None
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[0]


def downsample_if_needed(image, bboxes, masks=None, target_max_dim=DOWNSCALE_TARGET):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= target_max_dim:
        return image, bboxes, masks, 1.0
    scale = target_max_dim / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"[INFO] Downsampling image: {w}x{h} -> {new_w}x{new_h} (scale={scale:.4f})")
    image_ds = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_bboxes = [[c * scale for c in bbox] for bbox in bboxes]
    masks_ds = None
    if masks:
        masks_ds = [cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST) for m in masks]
    return image_ds, scaled_bboxes, masks_ds, scale


def bgr_to_rgb(bgr):
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def nonwhite_mask_u8(patch_rgb, white_thresh=245):
    flat = patch_rgb.reshape(-1, 3)
    keep = (np.any(flat < white_thresh, axis=1)).astype(np.uint8) * 255
    return keep.reshape(patch_rgb.shape[0], patch_rgb.shape[1])


_SUP_MAP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
            "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
            "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
            "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ",
            "M": "ᴹ", "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "T": "ᵀ", "U": "ᵁ", "V": "ⱽ", "W": "ᵂ"}
_SUB_MAP = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
            "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
            "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
            "v": "ᵥ", "x": "ₓ"}


def _to_sup(s): return "".join(_SUP_MAP.get(ch, ch) for ch in str(s)) if s else ""


def _to_sub(s): return "".join(_SUB_MAP.get(ch, ch) for ch in str(s)) if s else ""


def symbol_to_terminal(sym: dict) -> str:
    if not sym: return "Unknown"

    # 1. Prioritize the complete symbol stitched by the vision model
    final = sym.get("final_symbol")

    # 2. If final exists and is valid, return it directly without overwriting!
    if final and str(final).lower() not in ["", "null", "unknown", "none"]:
        return final

    # 3. Only use mechanical stitching as a fallback if final extraction fails
    base = sym.get("base") or ""
    sup, sub = sym.get("superscript", ""), sym.get("subscript", "")

    if not base:
        return "Unknown"

    return f"{base}{_to_sup(sup)}{_to_sub(sub)}"


def augment_symbol_formats(sym: dict) -> dict:
    if not isinstance(sym, dict): return sym
    sym["terminal"] = symbol_to_terminal(sym)
    return sym

# =====================================================
# High-Precision Texture Feature & Color Quantization
# =====================================================
def compute_texture_feature(image_rgb, mask=None, radius=3, n_points=24):
    if image_rgb is None or image_rgb.size == 0: return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points + 2

    if mask is not None:
        if mask.dtype != bool: mask = mask.astype(bool)
        if np.sum(mask) == 0: return np.zeros(n_bins, dtype=np.float32)
        lbp_vals = lbp[mask]
    else:
        lbp_vals = lbp.ravel()

    # Normalize manually to sum to 1 without numpy's density, compatible with OpenCV
    hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return hist


def quantize_image_by_legends(image, legend_info):
    print("[INFO] Quantizing image (Cleaning noise & patterns)...")
    h, w, c = image.shape
    palette = [li["avg_color"] for li in legend_info]
    palette.append([255, 255, 255])
    palette.append([0, 0, 0])
    palette = np.array(palette, dtype=np.float32)
    flat_image = image.reshape(-1, 3).astype(np.float32)
    labels_list = []
    batch_size = 50000
    for i in range(0, flat_image.shape[0], batch_size):
        batch = flat_image[i: i + batch_size]
        dists = cdist(batch, palette, metric='euclidean')
        labels_list.append(np.argmin(dists, axis=1))
    all_labels = np.concatenate(labels_list)
    quantized_flat = palette[all_labels].astype(np.uint8)
    return quantized_flat.reshape(h, w, c)