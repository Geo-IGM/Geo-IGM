from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
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
from tools import bgr_to_rgb, nonwhite_mask_u8, compute_texture_feature

SAVE_DIR = "./output"
LEGEND_ITEM_DIR = os.path.join(SAVE_DIR, "legend_items")
WHITE_THRESH = 245

# =====================================================
# Core Logic
# =====================================================
def segment_main_map_by_felzenszwalb(image_rgb, main_mask, min_size=300, scale=800, sigma=0.5):
    print(f"[INFO] Segmenting by Felzenszwalb (scale={scale}, sigma={sigma}, min_size={min_size})...")

    # 1) Keep only the main map area, set non-main areas to white to reduce false segmentation
    work = image_rgb.copy()
    work[main_mask == 0] = [255, 255, 255]

    # 2) skimage's felzenszwalb is more suitable for float images
    work_float = img_as_float(work)

    # 3) Felzenszwalb segmentation, output label map
    seg = felzenszwalb(work_float, scale=scale, sigma=sigma, min_size=min_size)

    regions = []
    vis = image_rgb.copy()

    # 4) Iterate each label and extract contours
    unique_labels = np.unique(seg)
    for lab in unique_labels:
        mask = (seg == lab).astype(np.uint8) * 255

        # Restrict within the main map area
        mask = cv2.bitwise_and(mask, mask, mask=main_mask)

        if cv2.countNonZero(mask) < min_size:
            continue

        # Optional: perform morphological cleaning
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

    # Only check right and bottom neighbors to avoid duplication
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

        # May be Polygon or MultiPolygon
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

def extract_legend_items(image_rgb, legends_dict, save_dir=LEGEND_ITEM_DIR):
    """Note: The input image_rgb here must be the high-resolution original image to get the clearest legend"""
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

        # Important Update: Extract standard legend features using multi-scale LBP directly on the high-resolution original image
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