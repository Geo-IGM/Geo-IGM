import os
import json
import re
import cv2
import base64
import requests
import numpy as np
import matplotlib.image as mpimg
import skimage.io as sio

# =====================================================
# 1. Basic experiment configuration
# =====================================================
SAVE_DIR = "output"
LEGEND_ITEM_DIR = os.path.join(SAVE_DIR, "legend_items")

# Core experiment settings
API_URL = "YOUR_API_URL"  # Replace with your actual API endpoint
MODEL_NAME = "gpt-4o"  # Model name, e.g., "gpt-4o", "gemini-2.5-pro"
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")  # Read from env or fallback to hardcoded

HEADERS = {
    "Authorization": f"Bearer {API_KEY}" if "gpt" in MODEL_NAME.lower() else API_KEY,
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
# 2. LLM API interface
# =====================================================
def image_to_base64(image_array_rgb):
    """Convert NumPy image array to Base64 string"""
    img_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def llm_call(image_crop_rgb, prompt):
    """Wrapper for LLM API call"""
    b64_image = image_to_base64(image_crop_rgb)

# =====================================================
# 3. Result parsing and hallucination guard (unchanged)
# =====================================================
_SUP_MAP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
            "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
            "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ"}
_SUB_MAP = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
            "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
            "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
            "v": "ᵥ", "x": "ₓ"}


def _to_sup(s): return "".join(_SUP_MAP.get(ch, ch) for ch in str(s)) if s else ""


def _to_sub(s): return "".join(_SUB_MAP.get(ch, ch) for ch in str(s)) if s else ""


def safe_parse_json(text):
    if not text: return None
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match: text = match.group(1)
        return json.loads(text)
    except Exception:
        try:
            return json.loads(text)
        except:
            return None


def is_null_geology_symbol(sym):
    def _is_null_like(val):
        return val is None or str(val).strip() == "" or str(val).lower() == "null"

    def _has_tofu_or_pua(s: str) -> bool:
        if not s: return False
        if "□" in s: return True
        return re.search(r"[\uE000-\uF8FF]", s) is not None

    if _is_null_like(sym): return True
    if isinstance(sym, str):
        s, s_low = sym.strip(), sym.strip().lower()
        if _has_tofu_or_pua(s) or s_low in {"unknown", "unrecognized", "n/a", "na"} or "未识别" in s or "" in s:
            return True
        return False
    if isinstance(sym, dict):
        for _, v in sym.items():
            if isinstance(v, str) and _has_tofu_or_pua(v.strip()): return True
        cand = [sym.get(k) for k in ["final_symbol", "terminal", "base", "superscript", "subscript", "final"] if
                sym.get(k)]
        if all(_is_null_like(v) or (isinstance(v, str) and v.strip() == "") for v in cand): return True
    return False


def _ensure_text_fallback(txt):
    if isinstance(txt, dict) and "legend_text" in txt: return txt
    return {"legend_text": "Unknown"}


def _ensure_symbol_fallback(sym):
    if isinstance(sym, dict) and ("base" in sym or "final_symbol" in sym): return sym
    return {"final_symbol": "Unknown", "base": None, "superscript": None, "subscript": None}


def augment_symbol_formats(sym: dict) -> dict:
    if not isinstance(sym, dict): return sym
    final = sym.get("final_symbol")
    if final and str(final).lower() not in ["", "null", "unknown", "none"]:
        sym["terminal"] = final
    else:
        base = sym.get("base") or ""
        sup, sub = sym.get("superscript", ""), sym.get("subscript", "")
        sym["terminal"] = f"{base}{_to_sup(sup)}{_to_sub(sub)}" if base else "Unknown"
    return sym


# =====================================================
# 4. Legend patch extraction
# =====================================================
def extract_legend_items(image_rgb, legends_dict, save_dir=LEGEND_ITEM_DIR):
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

        color_path = os.path.join(save_dir, f"legend_color_{idx}.png")
        text_path = os.path.join(save_dir, f"legend_text_{idx}.png")
        cv2.imwrite(color_path, cv2.cvtColor(color_patch, cv2.COLOR_RGB2BGR))
        cv2.imwrite(text_path, cv2.cvtColor(text_patch, cv2.COLOR_RGB2BGR))

        legend_info.append({
            "id": int(idx),
            "color_img": color_path,
            "text_img": text_path,
            "llm_text": None,
            "llm_symbol": None
        })
    return sorted(legend_info, key=lambda x: x["id"])


# =====================================================
# 5. Main pipeline
# =====================================================
def run_evaluation(img_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[STEP 1] Loading {img_path}...")
    image = sio.imread(img_path)
    if image.shape[-1] == 4: image = image[..., :3]

    print("[STEP 2] Simulating legend bounding boxes...")
    legends_dict = {
        "0": {"color_bndbox": [10, 10, 100, 50], "text_bndbox": [110, 10, 300, 50]},
    }

    print("[STEP 3] Extracting text & color patches...")
    legend_info = extract_legend_items(image, legends_dict)

    cache_path = os.path.join(SAVE_DIR, f"cache_{MODEL_NAME}.json")
    cached_map = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf8") as f:
                for item in json.load(f):
                    if isinstance(item, dict) and "id" in item:
                        cached_map[item["id"]] = item
        except Exception:
            pass

    print(f"[STEP 4] Processing via {MODEL_NAME} API (Total: {len(legend_info)})...")
    for li in legend_info:
        lid = li["id"]
        cached_item = cached_map.get(lid, {"id": lid, "llm_text": None, "llm_symbol": None})
        li["llm_text"] = cached_item.get("llm_text")
        li["llm_symbol"] = cached_item.get("llm_symbol")

        need_text = li["llm_text"] is None
        need_symbol = is_null_geology_symbol(li["llm_symbol"])

        if need_text or need_symbol:
            print(f"  > Calling {MODEL_NAME} for Legend {lid}... (text={need_text}, symbol={need_symbol})")

            if need_text:
                img_text = mpimg.imread(li["text_img"])
                if img_text.dtype != np.uint8: img_text = (img_text * 255).astype(np.uint8)
                t_raw = llm_call(img_text, LEGEND_TEXT_PROMPT)
                li["llm_text"] = _ensure_text_fallback(safe_parse_json(t_raw))

            if need_symbol:
                img_color = mpimg.imread(li["color_img"])
                if img_color.dtype != np.uint8: img_color = (img_color * 255).astype(np.uint8)
                s_raw = llm_call(img_color, GEOLOGY_PROMPT_TEMPLATE)
                li["llm_symbol"] = augment_symbol_formats(_ensure_symbol_fallback(safe_parse_json(s_raw)))

            cached_map[lid] = {"id": lid, "llm_text": li["llm_text"], "llm_symbol": li["llm_symbol"]}
            with open(cache_path, "w", encoding="utf8") as f:
                json.dump(list(cached_map.values()), f, ensure_ascii=False, indent=2)

    out_filename = f"{MODEL_NAME}-result.txt"
    out_path = os.path.join(SAVE_DIR, out_filename)

    print(f"[STEP 5] Exporting test results to: {out_filename}")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(legend_info, f, ensure_ascii=False, indent=4)

    print("[DONE] Extraction completed.")


if __name__ == "__main__":
    test_img = "sample_cgs.jpg" # Set your image path here
    if not os.path.exists(test_img):
        print(f"❌ Error: Image not found at {test_img}")
    else:
        run_evaluation(test_img)
