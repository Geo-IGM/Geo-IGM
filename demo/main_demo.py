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
# Configuration Parameters
# =====================================================
DOWNSCALE_TARGET = 2500

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
# Prompt
# =====================================================
GEOLOGY_PROMPT_TEMPLATE = """【角色与核心指令】
你是一个超高精度的“视觉空间与排版解析器”，而非传统地质学家。你的唯一目标是：完全按照裁剪图像中地质符号的物理外观进行提取，严格做到“所见即所得”。

【防幻觉红线 (极其重要)】：
你必须完全屏蔽并忽略地质学命名惯例（如代、纪、世等地层常识）。严禁基于地质语义进行脑补、推断或强行将平齐的字符拆分为上下标。你的提取行为必须 100% 依赖图像中字符的绝对空间位置和相对字号大小。

【严格提取规则】
1. 主体 (base)：图像中字号最大或最左侧的字母确立了“基准线”。如果后续字符（数字或字母）的底端与此基准线平齐，且字号无明显缩小，强制将它们全部归入“主体”。
2. 上标 (superscript)：【仅当】某字符的底部边缘明显高于主体的水平中心线，且字号明显较小时。否则强制归入主体。若无，设为 "null"。
3. 下标 (subscript)：【仅当】某字符的顶部边缘明显低于主体的水平中心线，且字号明显较小时。否则强制归入主体。若无，设为 "null"。
4. 兜底防线：若对字符的垂直空间偏移置信度低于 80%（如字符看起来平齐），极度保守地将它们全部合并提取到“主体 (base)”中。绝对不允许为了凑齐标准地质格式而强行拆分连续字符。

【任务】
解析图例色块内部的地质符号，并将其解析为结构化数据。

【输出】仅输出有效的 JSON 对象，不要包含 ```json 等 Markdown 标记或任何解释文本：
{
  "base": "主体符号（严格依据视觉提取）", 
  "superscript": "上标（若无则为 null）", 
  "subscript": "下标（若无则为 null）",
  "final_symbol": "严格按照图像视觉拼接的完整符号字符串（如无上下标则与base一致）", 
  "confidence": 0.95
}"""

LEGEND_TEXT_PROMPT = """【角色与任务】
你是一名高精度的地质图例文字 OCR 与语义提取专家。
请精准识别图例右侧（或色块旁边）的文字描述，提取其中的地质单位说明（如地层名称、岩性描述、时代、构造特征等）。
要求：忠于原图文字，准确反映地层时代与岩性信息，去除多余的噪点字符或背景干扰。

【输出】仅输出有效的 JSON 对象，不要包含 ```json 等 Markdown 标记或任何解释文本：
{ 
  "legend_text": "提取出的完整地质描述文字", 
  "confidence": 0.95 
}"""


# =====================================================
# Basic Utility Functions
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

    final = sym.get("final_symbol")

    if final and str(final).lower() not in ["", "null", "unknown", "none"]:
        return final

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


# =====================================================
# Core Logic
# =====================================================
def segment_main_map_by_felzenszwalb(image_rgb, main_mask, min_size=300, scale=800, sigma=0.5):
    print(f"[INFO] Segmenting by Felzenszwalb (scale={scale}, sigma={sigma}, min_size={min_size})...")

    # 1) Keep only main map area, set background to white to reduce errors
    work = image_rgb.copy()
    work[main_mask == 0] = [255, 255, 255]

    # 2) Felzenszwalb works better with float images
    work_float = img_as_float(work)

    # 3) Run segmentation
    seg = felzenszwalb(work_float, scale=scale, sigma=sigma, min_size=min_size)

    regions = []
    vis = image_rgb.copy()

    # 4) Extract contours for each segment
    unique_labels = np.unique(seg)
    for lab in unique_labels:
        mask = (seg == lab).astype(np.uint8) * 255

        # Restrict to main map area
        mask = cv2.bitwise_and(mask, mask, mask=main_mask)

        if cv2.countNonZero(mask) < min_size:
            continue

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        if cv2.countNonZero(mask) < min_size:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_size:
                continue

            pts = c.squeeze()
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue

            poly = Polygon([(int(p[0]), int(p[1])) for p in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            regions.append({
                "id": len(regions),
                "contour": pts.tolist(),
                "area": int(area),
                "centroid": (float(poly.centroid.x), float(poly.centroid.y)),
                "matched_legend_id": None
            })

    cv2.drawContours(vis, [np.array(r["contour"]) for r in regions], -1, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(SAVE_DIR, "vis_seg_felzenszwalb.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Generated {len(regions)} regions by Felzenszwalb.")
    return regions
def build_adjacency(regions, shape):
    label_map = np.full(shape, -1, dtype=np.int32)

    for r in regions:
        contour = np.array(r["contour"], dtype=np.int32)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        cv2.drawContours(label_map, [contour], -1, int(r["id"]), thickness=-1)

    adjacency = {r["id"]: set() for r in regions}
    h, w = label_map.shape

    for y in range(h - 1):
        for x in range(w - 1):
            a = label_map[y, x]
            b = label_map[y, x + 1]
            c = label_map[y + 1, x]

            if a >= 0 and b >= 0 and a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)

            if a >= 0 and c >= 0 and a != c:
                adjacency[a].add(c)
                adjacency[c].add(a)

    return adjacency

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def merge_regions_by_label(regions, adjacency, min_match_score=0):
    visited = set()
    merged = []
    id_to_region = {r["id"]: r for r in regions}

    for r in regions:
        rid = r["id"]
        if rid in visited:
            continue

        stack = [rid]
        group = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)

            for nb in adjacency.get(cur, []):
                if nb in visited:
                    continue
                if (
                        id_to_region[nb].get("matched_legend_id") == r.get("matched_legend_id")
                        and id_to_region[nb].get("match_score", 0) >= min_match_score
                ):
                    stack.append(nb)
        polys = []
        total_area = 0

        for gid in group:
            cnt = np.array(id_to_region[gid]["contour"], dtype=np.int32)
            if cnt.ndim != 2 or cnt.shape[0] < 3:
                continue

            poly = Polygon(cnt)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            polys.append(poly)
            total_area += id_to_region[gid].get("area", 0)

        if not polys:
            continue

        merged_poly = unary_union(polys)

        if merged_poly.geom_type == "Polygon":
            geoms = [merged_poly]
        elif merged_poly.geom_type == "MultiPolygon":
            geoms = list(merged_poly.geoms)
        else:
            continue

        for g in geoms:
            coords = np.array(g.exterior.coords, dtype=np.int32)
            if coords.shape[0] < 3:
                continue

            merged.append({
                "id": len(merged),
                "contour": coords.tolist(),
                "area": int(g.area),
                "centroid": (float(g.centroid.x), float(g.centroid.y)),
                "matched_legend_id": r.get("matched_legend_id"),
                "match_score": max(
                    [id_to_region[gid].get("match_score", 0) for gid in group],
                    default=0
                ),
                "region_color_rgb": r.get("region_color_rgb"),
                "top_matches": r.get("top_matches")
            })

    return merged
def save_regions_vis(image_rgb, regions, out_name="vis_regions.png", color=(0, 255, 0), thickness=2):
    vis = image_rgb.copy()
    contours = []
    for r in regions:
        cnt = np.array(r["contour"], dtype=np.int32)
        if cnt.ndim == 2 and cnt.shape[0] >= 3:
            contours.append(cnt)
    if contours:
        cv2.drawContours(vis, contours, -1, color, thickness)
    cv2.imwrite(os.path.join(SAVE_DIR, out_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
# Extract legend items from high-resolution original image
def extract_legend_items(image_rgb, legends_dict, save_dir=LEGEND_ITEM_DIR):
    """Note: input image must be high-res original for clear legend extraction"""
    os.makedirs(save_dir, exist_ok=True)
    legend_info = []
    h, w = image_rgb.shape[:2]
    for idx, info in legends_dict.items():
        x0, y0, x1, y1 = map(int, info["color_bndbox"])
        tx0, ty0, tx1, ty1 = map(int, info["text_bndbox"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        tx0, ty0, tx1, ty1 = max(0, tx0), max(0, ty0), min(w, tx1), min(h, ty1)
        if x1 <= x0 or y1 <= y0: continue

        color_patch = image_rgb[y0:y1, x0:x1].copy()
        text_patch = image_rgb[ty0:ty1, tx0:tx1].copy()
        mean_rgb = bgr_to_rgb(info["color"]) if info.get("color") else [127, 127, 127]
        mask_u8 = nonwhite_mask_u8(color_patch, WHITE_THRESH)

        lbp_hist = compute_texture_feature(color_patch, mask=(mask_u8 == 255), radius=3, n_points=24)

        color_path = os.path.join(save_dir, f"legend_color_{idx}.png")
        text_path = os.path.join(save_dir, f"legend_text_{idx}.png")
        cv2.imwrite(color_path, cv2.cvtColor(color_patch, cv2.COLOR_RGB2BGR))
        cv2.imwrite(text_path, cv2.cvtColor(text_patch, cv2.COLOR_RGB2BGR))

        legend_info.append({
            "id": int(idx),
            "avg_color": mean_rgb,
            "lbp": lbp_hist,
            "color_name": info.get("color_name", "unknown"),
            "color_img": color_path,
            "text_img": text_path,
            "gemini_text": None,
            "gemini_symbol": None
        })
    return sorted(legend_info, key=lambda x: x["id"])

# =====================================================
# High-Resolution Original Image Projection Matching
# =====================================================
def region_avg_rgb(image_rgb, contour_pts):
    contour = np.array(contour_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    mask_local = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask_local, [shifted_contour], -1, 255, -1)
    mask_local = cv2.erode(mask_local, np.ones((3, 3), np.uint8), iterations=2)
    crop = image_rgb[y:y + h, x:x + w]
    if np.sum(mask_local == 255) < 10:
        mask_local = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_local, [shifted_contour], -1, 255, -1)
    pixels = crop[mask_local == 255]
    if len(pixels) == 0: return [255, 255, 255], crop, mask_local
    s = np.sum(pixels, axis=1)
    valid = pixels[(s > 50) & (s < 720)]
    if len(valid) == 0: valid = pixels
    return np.median(valid, axis=0).astype(int).tolist(), crop, mask_local


def match_regions_to_legend_highres(regions_ds, legend_info, image_rgb, scale, topk=5):
    print("[INFO] High-Res Matching with Text-Filtering & Bhattacharyya distance...")
    l_data = [{"id": li["id"], "rgb": np.array(li["avg_color"], dtype=np.float32), "lbp": li.get("lbp"), "meta": li} for
              li in legend_info]
    inv_scale = 1.0 / scale

    for r in regions_ds:
        ds_contour = np.array(r["contour"])
        orig_contour = (ds_contour * inv_scale).astype(np.int32)

        # 1. Crop high-res ROI
        reg_rgb, reg_crop, reg_mask = region_avg_rgb(image_rgb, orig_contour)
        reg_arr = np.array(reg_rgb, dtype=np.float32)

        # 2. Clean mask: remove dark text/lines inside the region
        gray_crop = cv2.cvtColor(reg_crop, cv2.COLOR_RGB2GRAY)
        clean_mask = (reg_mask == 255) & (gray_crop > 80)
        reg_lbp = compute_texture_feature(reg_crop, mask=clean_mask, radius=3, n_points=24)

        # 3. Calculate distance scores
        candidates = []
        for l in l_data:
            # Color distance (0 ~ 441)
            diff = reg_arr - l["rgb"]
            c_dist = np.sqrt(0.3 * diff[0] ** 2 + 0.59 * diff[1] ** 2 + 0.11 * diff[2] ** 2)
            c_score = min(c_dist / 300.0, 1.0)

            # Texture distance (Bhattacharyya, 0~1)
            t_score = 1.0
            if l["lbp"] is not None and reg_lbp is not None and np.sum(reg_lbp) > 0:
                t_score = cv2.compareHist(reg_lbp, l["lbp"], cv2.HISTCMP_BHATTACHARYYA)

            # Combined score: color 0.6, texture 0.4
            total_dist = (0.6 * c_score) + (0.4 * t_score)

            candidates.append({
                "lid": l["id"],
                "score": total_dist,
                "c_dist": round(c_score, 3),
                "t_dist": round(t_score, 3),
                "meta": l["meta"]
            })

        # Sort by smallest distance
        candidates.sort(key=lambda x: x["score"])
        best = candidates[0]

        r["matched_legend_id"] = best["lid"]
        # Convert distance to confidence score 0~100
        r["match_score"] = round(max((1.0 - best["score"]) * 100.0, 0), 2)
        r["region_color_rgb"] = reg_rgb
        r["top_matches"] = [
            {"id": c["lid"], "score": round((1.0 - c["score"]) * 100, 2), "c": c["c_dist"], "t": c["t_dist"]} for c in
            candidates[:3]]

    return regions_ds

# =====================================================
# Export Functions
# =====================================================
# Export regions for UI usage (scale restored)
def export_regions_ui(regions, scale=1.0, out_json="regions_ui.json"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, out_json)
    inv_scale = 1.0 / scale
    rows = []
    for r in regions:
        raw_cnt = np.array(r.get("contour"))
        real_cnt = (raw_cnt * inv_scale).tolist() if len(raw_cnt) > 0 else []
        cx, cy = r.get("centroid")
        rows.append({
            "id": int(r["id"]),
            "contour": real_cnt,
            "centroid": (cx * inv_scale, cy * inv_scale),
            "area": int(r.get("area", 0) * (inv_scale * inv_scale)),
            "matched_legend_id": r.get("matched_legend_id"),
            "match_score": r.get("match_score"),
            "geo": r.get("geo"),
            "region_color_rgb": r.get("region_color_rgb"),
            "top_matches": r.get("top_matches")
        })
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] UI JSON exported.")


def export_regions(regions, legend_info, out_csv="region_matches_symbols.csv"):
    rows = []

    def ensure_dict(data):
        if isinstance(data, dict): return data
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict): return data[0]
        return {}

    for r in regions:
        geo = r.get("geo") or {}
        sym = ensure_dict(geo.get("symbol"))
        rows.append({
            "region_id": int(r["id"]),
            "matched_legend_id": int(r["matched_legend_id"]),
            "match_score": float(r.get("match_score", 0)),
            "symbol_final": sym.get("final") or sym.get("final_symbol") or sym.get("terminal"),
            "legend_text": geo.get("unit_name"),
            "legend_color_name": geo.get("legend_color_name"),
            "area_px": int(r.get("area", 0)),
        })

    for li in legend_info:
        sym = ensure_dict(li.get("gemini_symbol"))
        txt = ensure_dict(li.get("gemini_text"))
        rows.insert(0, {
            "region_id": -1 - int(li["id"]),
            "matched_legend_id": int(li["id"]),
            "match_score": None,
            "symbol_final": sym.get("final_symbol") or sym.get("terminal"),
            "legend_text": txt.get("legend_text"),
            "legend_color_name": li.get("color_name"),
            "area_px": -1
        })

    csv_path = os.path.join(SAVE_DIR, out_csv)
    headers = ["region_id", "matched_legend_id", "match_score", "symbol_final", "legend_text", "legend_color_name",
               "area_px"]
    try:
        with open(csv_path, "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] CSV exported successfully.")
    except Exception as e:
        print(f"[WARN] Failed to write CSV: {e}")


# =====================================================
# Main Pipeline
# =====================================================
def run(img_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[STEP 1] Loading {img_path}...")
    image = sio.imread(img_path)
    if image.shape[-1] == 4:
        image = image[..., :3]

    print("[STEP 2] Detecting components...")
    comp_det = map_component_detector()
    leg_det = map_legend_detector()
    comp = comp_det.detect(img_path)
    legends_dict = leg_det.detect(img_path)

    # Extract legend items from original high-res image
    print("[STEP 2.5] Extracting legend items on High-Res image...")
    legend_info = extract_legend_items(image, legends_dict)

    # Gemini API call with cache
    cache_path = os.path.join(SAVE_DIR, "legend_gemini_cache.json")
    cached_map = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf8") as f:
                raw_data = json.load(f)

            cached_map = {}
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id", item.get("legend_id"))
                if item_id is None:
                    continue

                if "gemini_symbol" not in item and "symbol" in item:
                    item["gemini_symbol"] = item["symbol"]

                cached_map[item_id] = item
        except Exception:
            pass

    print(f"[STEP 3] Processing Gemini API (Total: {len(legend_info)})...")
    for li in legend_info:
        lid = li["id"]
        if lid in cached_map:
            cached_item = cached_map[lid]
            li["gemini_text"] = cached_item.get("gemini_text")
            li["gemini_symbol"] = cached_item.get("gemini_symbol")
        else:
            cached_item = {"id": lid, "gemini_text": None, "gemini_symbol": None}

        need_text = (li.get("gemini_text") is None)
        need_symbol = is_null_geology_symbol(li.get("gemini_symbol"))

        if need_text or need_symbol:
            print(f"  > Calling Gemini for Legend {lid}... (text={need_text}, symbol={need_symbol})")

            if need_text:
                img_text = mpimg.imread(li["text_img"])
                if img_text.dtype != np.uint8:
                    img_text = (img_text * 255).astype(np.uint8)
                t_raw = gemini_call(img_text, LEGEND_TEXT_PROMPT)
                li["gemini_text"] = safe_parse_json(t_raw)
                li["gemini_text"] = _ensure_text_fallback(li["gemini_text"])

            if need_symbol:
                img_color = mpimg.imread(li["color_img"])
                if img_color.dtype != np.uint8:
                    img_color = (img_color * 255).astype(np.uint8)
                s_raw = gemini_call(img_color, GEOLOGY_PROMPT_TEMPLATE)
                li["gemini_symbol"] = safe_parse_json(s_raw)
                li["gemini_symbol"] = _ensure_symbol_fallback(li["gemini_symbol"])

            li["gemini_symbol"] = augment_symbol_formats(li.get("gemini_symbol"))
            cached_item["gemini_text"] = li.get("gemini_text")
            cached_item["gemini_symbol"] = li.get("gemini_symbol")
            cached_map[lid] = cached_item

            with open(cache_path, "w", encoding="utf8") as f:
                json.dump(list(cached_map.values()), f, ensure_ascii=False, indent=2)

        else:
            li["gemini_text"] = _ensure_text_fallback(li.get("gemini_text"))
            li["gemini_symbol"] = _ensure_symbol_fallback(li.get("gemini_symbol"))
            li["gemini_symbol"] = augment_symbol_formats(li.get("gemini_symbol"))

    # Prepare main map mask and downscale image
    main_bbox = get_component_bbox(comp, "main_map")
    legend_bbox = get_component_bbox(comp, "legend")

    if not main_bbox:
        main_bbox = [0, 0, image.shape[1], image.shape[0]]

    all_bboxes = [main_bbox]
    if legend_bbox:
        all_bboxes.append(legend_bbox)

    for v in legends_dict.values():
        all_bboxes.extend([v["color_bndbox"], v["text_bndbox"]])

    main_mask = bbox_to_mask(image.shape, main_bbox, pad=10)

    image_ds, scaled_bboxes, masks_ds, scale = downsample_if_needed(
        image, all_bboxes, [main_mask]
    )
    main_mask_ds = masks_ds[0]

    # Main map preprocessing and grid line removal
    print("[STEP 3.5] Extracting and inpainting grid lines...")

    # Edge detection
    gray_ds = cv2.cvtColor(image_ds, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_ds, 50, 150)

    # Morphological detection for horizontal/vertical lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    mask_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    mask_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

    grid_mask = cv2.bitwise_or(mask_h, mask_v)

    grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)

    image_ds = cv2.inpaint(image_ds, grid_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    clean_image = cv2.bilateralFilter(image_ds, 9, 75, 75)

    # Color quantization using legend palette
    quantized_img = quantize_image_by_legends(clean_image, legend_info)

    # Run segmentation
    # -------------------------------------------------
    print("[STEP 4] Segmentation (Felzenszwalb Mode)...")
    regions = segment_main_map_by_felzenszwalb(
        quantized_img,
        main_mask_ds,
        min_size=400,
        scale=1400,
        sigma=0.8
    )

    # Visualize initial segmentation
    save_regions_vis(
        image_ds,
        regions,
        out_name="vis_felz_regions_before_merge.png",
        color=(255, 0, 0),
        thickness=1
    )

    # High-res feature matching
    print("[STEP 5] Matching (High-Res Map projection)...")
    regions = match_regions_to_legend_highres(regions, legend_info, image, scale)

    l_map = {li["id"]: li for li in legend_info}

    # Add geological info to regions
    for r in regions:
        lid = r.get("matched_legend_id")
        li = l_map.get(lid)
        if li:
            r["geo"] = {
                "unit_name": (li.get("gemini_text") or {}).get("legend_text"),
                "symbol": (li.get("gemini_symbol") or {}),
                "legend_color_name": li.get("color_name"),
                "legend_color_rgb": li.get("avg_color")
            }

    # -------------------------------------------------
    # Build region adjacency graph
    print("[STEP 5.5] Building adjacency...")
    adjacency = build_adjacency(regions, main_mask_ds.shape)

    # -------------------------------------------------
    # Merge adjacent regions with same label
    print("[STEP 5.6] Merging adjacent same-class regions...")
    regions = merge_regions_by_label(
        regions,
        adjacency,
        min_match_score=0
    )

    # Update geo info after merging
    for r in regions:
        lid = r.get("matched_legend_id")
        li = l_map.get(lid)
        if li:
            r["geo"] = {
                "unit_name": (li.get("gemini_text") or {}).get("legend_text"),
                "symbol": (li.get("gemini_symbol") or {}),
                "legend_color_name": li.get("color_name"),
                "legend_color_rgb": li.get("avg_color")
            }

    # Visualize merged regions
    save_regions_vis(
        image_ds,
        regions,
        out_name="vis_felz_regions_after_merge.png",
        color=(0, 255, 0),
        thickness=2
    )

    # -------------------------------------------------
    # Export results
    print("[STEP 6] Exporting...")
    export_regions_ui(regions, scale=scale)
    export_regions(regions, legend_info)

    print("[DONE] Success.")

if __name__ == "__main__":
    img_path = r"D:\Desktop\IGM\data\maps\sample_cgs.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found {img_path}")
    else:
        run(img_path)