# coding: utf-8
import json
import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage import io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

# ================== Font Definition ==================
FONT_UI = FontProperties(family=["Microsoft YaHei", "SimHei", "DejaVu Sans"])
FONT_SYMBOL = FontProperties(family=["DejaVu Sans", "Microsoft YaHei", "SimHei"])

plt.rcParams["axes.unicode_minus"] = False


# ================== Image Loading ==================
def load_rgb(img_path: str) -> np.ndarray:
    img = sio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


# ================== Build Region Paths ==================
def build_paths(regions):
    items = []
    for r in regions:
        contour = r.get("contour")
        if not contour or len(contour) < 3:
            continue

        pts = np.asarray(contour, dtype=np.float32)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()

        verts = np.vstack([pts, pts[0]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]
        path = Path(verts, codes)

        items.append({
            "id": r.get("id"),
            "bbox": (x0, y0, x1, y1),
            "path": path,
            "region": r
        })
    return items


# ================== Cache Compatibility Tools ==================
def normalize_cache_item(item: dict) -> Tuple[Optional[Any], dict]:
    if not isinstance(item, dict):
        return None, {}

    item_id = item.get("id", item.get("legend_id"))

    symbol_data = item.get("gemini_symbol")
    if not isinstance(symbol_data, dict):
        symbol_data = item.get("symbol")

    if isinstance(symbol_data, list):
        symbol_data = next((x for x in symbol_data if isinstance(x, dict)), {})
    elif not isinstance(symbol_data, dict):
        symbol_data = {}

    normalized = dict(item)
    normalized["id"] = item_id
    normalized["gemini_symbol"] = symbol_data

    return item_id, normalized


def pick_symbol_text(symbol_data) -> Optional[str]:
    if isinstance(symbol_data, list):
        for entry in symbol_data:
            text = pick_symbol_text(entry)
            if text:
                return text
        return None

    if isinstance(symbol_data, dict):
        for k in ("terminal", "final_symbol", "final", "text", "base"):
            v = symbol_data.get(k)
            if isinstance(v, str) and v.strip() and v.strip() != "Unknown":
                return v.strip()
        return None

    if isinstance(symbol_data, str) and symbol_data.strip() and symbol_data.strip() != "Unknown":
        return symbol_data.strip()

    return None


# ================== Information Extraction ==================
def get_region_info(r: dict, cache_map: dict) -> dict:
    geo = r.get("geo") or {}
    matched_id = r.get("matched_legend_id")
    score = r.get("match_score")
    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"

    area = r.get("area", 0)
    cx, cy = r.get("centroid", (0.0, 0.0))

    cache_info = cache_map.get(matched_id, {})

    unit_name = None
    gemini_text = cache_info.get("gemini_text")
    if isinstance(gemini_text, dict):
        unit_name = gemini_text.get("legend_text")
    if not unit_name:
        unit_name = geo.get("unit_name") or "N/A"

    symbol_text = None
    sym = cache_info.get("gemini_symbol")
    symbol_text = pick_symbol_text(sym)
    if not symbol_text:
        symbol_text = pick_symbol_text(geo.get("symbol"))
    if not symbol_text:
        symbol_text = "Unknown"

    # Updated to English Keys
    return {
        "Region ID": r.get("id"),
        "Matched Legend ID": matched_id,
        "Matching Score": score_str,
        "Region Color RGB": r.get("region_color_rgb"),
        "Legend Color Name": geo.get("legend_color_name") or "N/A",
        "Legend RGB": geo.get("legend_color_rgb"),
        "Geological Description": unit_name,
        "Area (px^2)": f"{area:,}",
        "Center Coordinates": f"({cx:.1f}, {cy:.1f})",
        "Geological Symbol": symbol_text,
    }


# ================== Construct Info Box ==================
def build_info_box(r: dict, cache_map: dict, fontsize: int = 10):
    info = get_region_info(r, cache_map)

    textprops_ui = {
        "fontproperties": FONT_UI,
        "fontsize": fontsize,
        "color": "black",
    }
    textprops_symbol = {
        "fontproperties": FONT_SYMBOL,
        "fontsize": fontsize,
        "color": "blue",
        "weight": "bold",
    }

    rows = []

    # Updated to English Display Output
    normal_lines = [
        f"Region ID: {info['Region ID']}",
        f"Matched Legend ID: {info['Matched Legend ID']}",
        f"Matching Score: {info['Matching Score']}",
        f"Region Color RGB: {info['Region Color RGB']}",
        f"Legend Color Name: {info['Legend Color Name']}",
        f"Legend RGB: {info['Legend RGB']}",
        f"Geological Description: {info['Geological Description']}",
        f"Area (px^2): {info['Area (px^2)']}",
        f"Center Coordinates: {info['Center Coordinates']}",
    ]

    for line in normal_lines:
        rows.append(TextArea(line, textprops=textprops_ui))

    # Last line: Label + Blue symbol
    label_area = TextArea("Geological Symbol: ", textprops=textprops_ui)
    symbol_area = TextArea(info["Geological Symbol"], textprops=textprops_symbol)

    last_row = HPacker(
        children=[label_area, symbol_area],
        align="baseline",
        pad=0,
        sep=2
    )
    rows.append(last_row)

    vbox = VPacker(
        children=rows,
        align="left",
        pad=0,
        sep=2
    )
    return vbox


# ================== Main Viewer ==================
def launch_viewer(img_path: str, regions_ui_json: str, cache_json: str):
    if not os.path.exists(img_path):
        print(f"❌ Error: Cannot find image file {img_path}")
        return
    if not os.path.exists(regions_ui_json):
        print(f"❌ Error: Cannot find region data file {regions_ui_json}")
        return

    cache_map: Dict[Any, dict] = {}
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                cache_list = json.load(f)

            if isinstance(cache_list, dict):
                cache_iter = list(cache_list.values())
            elif isinstance(cache_list, list):
                cache_iter = cache_list
            else:
                cache_iter = []

            skipped = 0
            for item in cache_iter:
                item_id, normalized = normalize_cache_item(item)
                if item_id is None:
                    skipped += 1
                    continue
                cache_map[item_id] = normalized

            print(f"✅ Successfully loaded legend cache file: Found {len(cache_map)} latest records.")
            if skipped:
                print(f"⚠️ Skipped {skipped} unrecognizable cache records.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to read cache file ({e}), falling back to old data.")
    else:
        print(f"⚠️ Warning: Cache file {cache_json} not found, falling back to old data.")

    img = load_rgb(img_path)
    h, w = img.shape[:2]

    with open(regions_ui_json, "r", encoding="utf-8") as f:
        regions = json.load(f)

    items = build_paths(regions)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Geological Map Interactive Viewer (Dynamic Cache)")

    ax.imshow(img, origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")

    # Updated Interactive Prompt
    ax.set_title(
        "Scroll: Zoom | Left-Drag: Pan | Hover: View Region Info",
        fontproperties=FONT_UI
    )

    highlight_line, = ax.plot([], [], linewidth=2, color="red")

    is_panning = False
    pan_start = {}
    current_hit_id = None
    info_ab = None

    def set_highlight(contour):
        pts = np.asarray(contour)
        if len(pts) > 0:
            xs = np.append(pts[:, 0], pts[0, 0])
            ys = np.append(pts[:, 1], pts[0, 1])
            highlight_line.set_data(xs, ys)

    def clear_highlight():
        highlight_line.set_data([], [])

    def remove_info_box():
        nonlocal info_ab
        if info_ab is not None:
            info_ab.remove()
            info_ab = None

    def create_info_box(x, y, r):
        nonlocal info_ab
        remove_info_box()

        vbox = build_info_box(r, cache_map, fontsize=10)

        info_ab = AnnotationBbox(
            vbox,
            (x, y),
            xybox=(15, 15),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(0, 1),
            frameon=True,
            bboxprops=dict(
                boxstyle="round",
                fc="white",
                ec="black",
                alpha=0.95
            ),
            arrowprops=dict(arrowstyle="->")
        )
        ax.add_artist(info_ab)

    def on_press(event):
        nonlocal is_panning
        if event.button == 1 and event.inaxes == ax:
            is_panning = True
            pan_start["x"] = event.xdata
            pan_start["y"] = event.ydata
            pan_start["xlim"] = ax.get_xlim()
            pan_start["ylim"] = ax.get_ylim()
            remove_info_box()

    def on_release(event):
        nonlocal is_panning
        is_panning = False

    def on_drag(event):
        if not is_panning or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan_start["x"]
        dy = event.ydata - pan_start["y"]
        x0, x1 = pan_start["xlim"]
        y0, y1 = pan_start["ylim"]
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        fig.canvas.draw_idle()

    def on_move(event):
        nonlocal current_hit_id

        if is_panning or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        hit = None

        for it in items:
            x0, y0, x1, y1 = it["bbox"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                if it["path"].contains_point((x, y)):
                    hit = it
                    break

        hit_id = hit["id"] if hit else None

        if hit_id == current_hit_id:
            return

        current_hit_id = hit_id

        if not hit:
            remove_info_box()
            clear_highlight()
            fig.canvas.draw_idle()
            return

        r = hit["region"]
        set_highlight(r.get("contour"))
        create_info_box(x, y, r)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.xdata is None or event.ydata is None:
            return
        scale = 1.2 if event.button == "down" else 1 / 1.2
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[0] - cur_ylim[1]) * scale

        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[0] - event.ydata) / (cur_ylim[0] - cur_ylim[1])

        ax.set_xlim([
            event.xdata - new_w * (1 - relx),
            event.xdata + new_w * relx
        ])
        ax.set_ylim([
            event.ydata + new_h * (1 - rely),
            event.ydata - new_h * rely
        ])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_drag)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()


# ================== CLI ==================
def main():
    DEFAULT_IMG_PATH = "sample_cgs.jpg"
    DEFAULT_JSON_PATH = "regions_ui.json"
    DEFAULT_CACHE_PATH = "legend_gemini_cache.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=DEFAULT_IMG_PATH, help="Original geological map path")
    parser.add_argument("--regions", default=DEFAULT_JSON_PATH, help="Exported JSON path")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH, help="Gemini cache JSON path")
    args = parser.parse_args()

    print("========================================")
    print("🚀 Starting Geological Map Viewer...")
    print(f"🖼️ Image path: {args.img}")
    print(f"🗺️ Region path: {args.regions}")
    print(f"🧠 Cache path: {args.cache}")
    print("========================================")

    launch_viewer(args.img, args.regions, args.cache)


if __name__ == "__main__":
    main()
