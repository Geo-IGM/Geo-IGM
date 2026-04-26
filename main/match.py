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
from tools import compute_texture_feature

SAVE_DIR = "./output"

# =====================================================
# High-resolution original image projection matching mechanism
# Combines color difference and Bhattacharyya LBP feature matching
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

        # 1. Crop high-resolution ROI
        reg_rgb, reg_crop, reg_mask = region_avg_rgb(image_rgb, orig_contour)
        reg_arr = np.array(reg_rgb, dtype=np.float32)

        # 2. Critical fix: Clean mask, remove black text and lines inside ROI!
        gray_crop = cv2.cvtColor(reg_crop, cv2.COLOR_RGB2GRAY)
        # Assume text/lines are dark (gray < 80), background is light/colored.
        # Only extract LBP texture from non-dark regions!
        clean_mask = (reg_mask == 255) & (gray_crop > 80)
        reg_lbp = compute_texture_feature(reg_crop, mask=clean_mask, radius=3, n_points=24)

        # 3. Calculate scientific distance score
        candidates = []
        for l in l_data:
            # Color distance (0 ~ ~441)
            diff = reg_arr - l["rgb"]
            c_dist = np.sqrt(0.3 * diff[0] ** 2 + 0.59 * diff[1] ** 2 + 0.11 * diff[2] ** 2)
            # Normalize color distance to 0~1 range (max color difference ~300)
            c_score = min(c_dist / 300.0, 1.0)

            # Texture distance (using stable Bhattacharyya distance, 0~1 range, 0 = perfect match)
            t_score = 1.0
            if l["lbp"] is not None and reg_lbp is not None and np.sum(reg_lbp) > 0:
                t_score = cv2.compareHist(reg_lbp, l["lbp"], cv2.HISTCMP_BHATTACHARYYA)

            # Fusion score: color weight 0.6, texture weight 0.4
            total_dist = (0.6 * c_score) + (0.4 * t_score)

            candidates.append({
                "lid": l["id"],
                "score": total_dist,
                "c_dist": round(c_score, 3),
                "t_dist": round(t_score, 3),
                "meta": l["meta"]
            })

        # Select the one with minimum total distance (closer to 0 is better)
        candidates.sort(key=lambda x: x["score"])
        best = candidates[0]

        r["matched_legend_id"] = best["lid"]
        # Convert 0~1 distance to 0~100 matching confidence
        r["match_score"] = round(max((1.0 - best["score"]) * 100.0, 0), 2)
        r["region_color_rgb"] = reg_rgb
        r["top_matches"] = [
            {"id": c["lid"], "score": round((1.0 - c["score"]) * 100, 2), "c": c["c_dist"], "t": c["t_dist"]} for c in
            candidates[:3]]

    # Return corrected region information
    return regions_ds

# =====================================================
# Export logic
# =====================================================
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