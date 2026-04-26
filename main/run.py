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
from tools import safe_parse_json, is_null_geology_symbol,_ensure_text_fallback, _ensure_symbol_fallback, gemini_call, bbox_to_mask, get_component_bbox, downsample_if_needed, augment_symbol_formats, quantize_image_by_legends
from match import match_regions_to_legend_highres, export_regions_ui, export_regions
from segment import segment_main_map_by_felzenszwalb, build_adjacency, merge_regions_by_label, save_regions_vis, extract_legend_items

# =====================================================
# 参数配置
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
FALLBACK_GEMINI_KEY = "Enter api_key here"

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
  "confidence": 0.95"
}"""

# =====================================================
# Main Workflow
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

    # -------------------------------------------------
    # Extract legend from original unscaled image
    # -------------------------------------------------
    print("[STEP 2.5] Extracting legend items on High-Res image...")
    legend_info = extract_legend_items(image, legends_dict)

    # -------------------------------------------------
    # Gemini API Call (with cache)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Prepare main map bbox/mask and downscale large image
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Main map preprocessing and grid removal (new module)
    # -------------------------------------------------
    print("[STEP 3.5] Extracting and inpainting grid lines...")

    # 1. Edge detection to extract full-image high-frequency gradients
    gray_ds = cv2.cvtColor(image_ds, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_ds, 50, 150)

    # 2. Morphological filtering: extract horizontal and vertical through lines
    # Note: (40, 1) and (1, 40) detect continuous lines of at least 40 pixels.
    # Reduce to 30 if grid lines are broken.
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    mask_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    mask_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

    # 3. Merge horizontal and vertical features to generate complete grid mask
    grid_mask = cv2.bitwise_or(mask_h, mask_v)

    # 4. Dilate mask: expand 1-2 pixels to cover anti-aliased edges of grid lines
    grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)

    # 5. Image inpainting using Telea algorithm: erase grid lines using surrounding colors
    image_ds = cv2.inpaint(image_ds, grid_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # After inpainting, geological patches disconnected by grid are reconnected
    clean_image = cv2.bilateralFilter(image_ds, 9, 75, 75)

    # Quantize downscaled main map using high-res legend colors
    quantized_img = quantize_image_by_legends(clean_image, legend_info)

    # -------------------------------------------------
    # Felzenszwalb Segmentation
    # -------------------------------------------------
    print("[STEP 4] Segmentation (Felzenszwalb Mode)...")
    regions = segment_main_map_by_felzenszwalb(
        quantized_img,
        main_mask_ds,
        min_size=400,
        scale=1400,
        sigma=0.8
    )

    # Visualize initial segmentation results
    save_regions_vis(
        image_ds,
        regions,
        out_name="vis_felz_regions_before_merge.png",
        color=(255, 0, 0),
        thickness=1
    )

    # -------------------------------------------------
    # High-res original image projection matching
    # -------------------------------------------------
    print("[STEP 5] Matching (High-Res Map projection)...")
    regions = match_regions_to_legend_highres(regions, legend_info, image, scale)

    l_map = {li["id"]: li for li in legend_info}

    # Add geo info for small regions
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
    # New: Build adjacency relationship
    # -------------------------------------------------
    print("[STEP 5.5] Building adjacency...")
    adjacency = build_adjacency(regions, main_mask_ds.shape)

    # -------------------------------------------------
    # New: Merge adjacent regions with same class
    # -------------------------------------------------
    print("[STEP 5.6] Merging adjacent same-class regions...")
    regions = merge_regions_by_label(
        regions,
        adjacency,
        min_match_score=0
    )

    # Re-add geo info after merging
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

    # Visualize after merging
    save_regions_vis(
        image_ds,
        regions,
        out_name="vis_felz_regions_after_merge.png",
        color=(0, 255, 0),
        thickness=2
    )

    # -------------------------------------------------
    # Export
    # -------------------------------------------------
    print("[STEP 6] Exporting...")
    export_regions_ui(regions, scale=scale)
    export_regions(regions, legend_info)

    print("[DONE] Success.")

if __name__ == "__main__":
    img_path = "sample_cgs.jpg" #Your image path
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found {img_path}")
    else:
        run(img_path)