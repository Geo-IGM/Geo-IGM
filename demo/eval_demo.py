# coding: utf-8
import tkinter as tk
from tkinter import simpledialog
import os
import json
import argparse
import math
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from skimage import io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =========================================================
# Basic utilities
# =========================================================
def load_rgb(img_path: str) -> np.ndarray:
    """Load RGB image, handle grayscale and RGBA inputs"""
    img = sio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def save_rgb(path: str, img_rgb: np.ndarray):
    """Save RGB image to file, create directory if needed"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def ensure_int_contour(contour) -> np.ndarray:
    """Convert contour to integer numpy array, validate minimum points"""
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim != 2 or cnt.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.int32)
    return cnt.astype(np.int32)


# =========================================================
# Human-in-the-loop interactive review annotator
# Clean visual version with thin black highlight lines
# =========================================================
class ReviewAnnotator:
    def __init__(self, img_path: str, save_json: str, pred_json: str = None):
        self.img_path = img_path
        self.save_json = save_json
        self.img = load_rgb(img_path)
        self.base_img = self.img.copy()

        self.regions: List[Dict[str, Any]] = []
        self.next_id = 0

        # Mode: 'edit' | 'draw'
        self.mode = 'edit'
        self.current_points: List[List[float]] = []

        # Interaction state
        self.hovered_region_idx = -1
        self.hovered_vertex = None
        self.dragging_vertex = None

        # Visual toggle
        self.show_all_labels = False  # Hide all text labels by default

        load_path = None
        if os.path.exists(save_json):
            load_path = save_json
            print(f"[INFO] Resuming from existing GT: {save_json}")
        elif pred_json and os.path.exists(pred_json):
            load_path = pred_json
            print(f"[INFO] Loading predictions as baseline: {pred_json}")

        if load_path:
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if isinstance(old, list):
                    self.regions = old
                    if len(self.regions) > 0:
                        self.next_id = max(int(r.get("id", 0)) for r in self.regions) + 1
            except Exception as e:
                print(f"[WARN] Failed to load json file: {e}")

        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.redraw()
        self.print_help()

    def print_help(self):
        """Print keyboard shortcuts and usage instructions"""
        print("\n" + "=" * 60)
        print("💡 Interactive Review & Ground Truth Editor (Clean Version)")
        print("Global Shortcuts")
        print("  t : Toggle global category label display (hidden by default)")
        print("  s : Save current results")
        print("  q : Save and exit")
        print("  n : Enter [New Drawing] mode")
        print("  esc: Return to [Review/Edit] mode")
        print("\nReview/Edit Mode (Default)")
        print("  Hover over region : Auto-highlight (thin black line), show vertices and category")
        print("  Left-click + drag vertex : Modify polygon boundary")
        print("  Hover + d : Delete region")
        print("  Hover + c : Modify region category (input in terminal)")
        print("\nNew Drawing Mode (Press n to enter)")
        print("  Left-click : Add vertex | Right-click : Undo last vertex | Enter : Finish")
        print("=" * 60 + "\n")

    def find_nearest_element(self, x, y) -> Tuple[int, Tuple[int, int]]:
        """Find hovered region and nearest vertex within search radius"""
        hovered_reg = -1
        nearest_vtx = None
        min_dist = float('inf')
        search_radius = 15.0

        for i, r in enumerate(self.regions):
            contour = np.array(r.get("contour", []))
            if contour.shape[0] < 3: continue

            if cv2.pointPolygonTest(contour.astype(np.float32), (x, y), False) >= 0:
                hovered_reg = i

            for j, pt in enumerate(contour):
                dist = math.hypot(pt[0] - x, pt[1] - y)
                if dist < search_radius and dist < min_dist:
                    min_dist = dist
                    nearest_vtx = (i, j)

        return hovered_reg, nearest_vtx

    def redraw(self):
        """Redraw the entire canvas with current annotations"""
        self.ax.clear()
        self.ax.imshow(self.base_img)

        mode_str = "DRAWING" if self.mode == 'draw' else "EDITING"
        self.ax.set_title(f"[{mode_str}] 'n':New | 'd':Del | 'c':Class | 't':Toggle Labels | Drag Vertices | 's':Save")

        for i, r in enumerate(self.regions):
            contour = ensure_int_contour(r.get("contour", []))
            if contour.shape[0] < 3: continue

            xs, ys = contour[:, 0], contour[:, 1]
            xs_cl, ys_cl = np.r_[xs, xs[0]], np.r_[ys, ys[0]]

            is_hovered = (i == self.hovered_region_idx) and (self.mode == 'edit')

            # --- Core visual optimization logic ---
            if is_hovered:
                # When hovered: thin black line, small black vertices, red text label
                self.ax.plot(xs_cl, ys_cl, color='black', linewidth=1.2)
                self.ax.scatter(xs, ys, color='black', s=12, alpha=0.8)

                cx, cy = np.mean(xs), np.mean(ys)
                label = str(r.get("matched_legend_id", r.get("label", r.get("id", "?"))))
                self.ax.text(cx, cy, label, color='white', fontsize=11, fontweight='bold',
                             ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))
            else:
                # When not hovered: thin green line, semi-transparent, no vertices
                self.ax.plot(xs_cl, ys_cl, color='lime', linewidth=1.0, alpha=0.6)

                # Show labels if global display is enabled
                if self.show_all_labels:
                    cx, cy = np.mean(xs), np.mean(ys)
                    label = str(r.get("matched_legend_id", r.get("label", r.get("id", "?"))))
                    self.ax.text(cx, cy, label, color='white', fontsize=8,
                                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

        # Highlight specifically hovered vertex (cyan to avoid confusion with black lines)
        if self.hovered_vertex and self.mode == 'edit':
            ri, pi = self.hovered_vertex
            px, py = self.regions[ri]["contour"][pi]
            self.ax.scatter([px], [py], color='cyan', s=80, edgecolors='white', zorder=5)

        # Draw region currently being created
        if self.mode == 'draw' and len(self.current_points) > 0:
            pts = np.array(self.current_points)
            self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1)
            self.ax.scatter(pts[:, 0], pts[:, 1], c='red', s=5)  # Size of points when adding new region

        self.fig.canvas.draw_idle()

    def on_move(self, event):
        """Handle mouse movement events"""
        if event.inaxes != self.ax or event.xdata is None: return

        if self.dragging_vertex and self.mode == 'edit':
            ri, pi = self.dragging_vertex
            self.regions[ri]["contour"][pi] = [event.xdata, event.ydata]
            self.redraw()
            return

        if self.mode == 'edit':
            h_reg, h_vtx = self.find_nearest_element(event.xdata, event.ydata)
            if h_reg != self.hovered_region_idx or h_vtx != self.hovered_vertex:
                self.hovered_region_idx = h_reg
                self.hovered_vertex = h_vtx
                self.redraw()

    def on_press(self, event):
        """Handle mouse button press events"""
        if event.inaxes != self.ax or event.xdata is None: return

        if self.mode == 'edit':
            if event.button == 1 and self.hovered_vertex:
                self.dragging_vertex = self.hovered_vertex

        elif self.mode == 'draw':
            if event.button == 1:
                self.current_points.append([float(event.xdata), float(event.ydata)])
                self.redraw()
            elif event.button == 3:
                if self.current_points:
                    self.current_points.pop()
                    self.redraw()

    def on_release(self, event):
        """Handle mouse button release events"""
        if event.button == 1 and self.dragging_vertex:
            self.dragging_vertex = None
            self.redraw()

    def on_key(self, event):
        """Handle keyboard press events"""
        if event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)
        elif event.key == "t":
            self.show_all_labels = not self.show_all_labels
            print(f"[INFO] Global label display {'enabled' if self.show_all_labels else 'disabled'}")
            self.redraw()
        elif event.key == "n":
            self.mode = 'draw'
            self.current_points = []
            print("[INFO] Switched to drawing mode (left-click to add points, Enter to finish)")
            self.hovered_region_idx, self.hovered_vertex = -1, None
            self.redraw()
        elif event.key == "escape":
            self.mode = 'edit'
            self.current_points = []
            print("[INFO] Exited drawing, returned to edit mode")
            self.redraw()

        if self.mode == 'edit':
            if event.key == "d" and self.hovered_region_idx != -1:
                removed = self.regions.pop(self.hovered_region_idx)
                print(f"[INFO] Deleted region: {removed.get('matched_legend_id', removed.get('id'))}")
                self.hovered_region_idx = -1
                self.redraw()

            elif event.key == "c" and self.hovered_region_idx != -1:
                r = self.regions[self.hovered_region_idx]
                old_cls = r.get('matched_legend_id', r.get('label', '?'))

                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()

                prompt_text = f"Current category is '{old_cls}'.\nEnter new category name (press Enter or Cancel to keep unchanged):"
                new_cls = simpledialog.askstring("Modify Category", prompt_text, parent=root)
                root.destroy()

                if new_cls is not None and new_cls.strip():
                    r['matched_legend_id'] = new_cls.strip()
                    r['label'] = new_cls.strip()
                    r['reviewed'] = True
                    print(f"[INFO] Category updated to: {new_cls.strip()}")
                    self.save()

                self.redraw()

        elif self.mode == 'draw':
            if event.key == "enter" and len(self.current_points) >= 3:
                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()
                new_cls = simpledialog.askstring(
                    "Enter Category",
                    "Polygon closed.\nEnter category name for this region (e.g., 'sandstone'):",
                    parent=root
                )
                root.destroy()

                if not new_cls or not new_cls.strip():
                    new_cls = "Unknown"

                self.regions.append({
                    "id": self.next_id,
                    "matched_legend_id": new_cls.strip(),
                    "label": new_cls.strip(),
                    "contour": self.current_points.copy(),
                    "reviewed": True
                })
                print(f"[INFO] Added new region id={self.next_id}, category={new_cls.strip()}")
                self.next_id += 1
                self.current_points = []
                self.mode = 'edit'
                self.redraw()
                self.save()

    def save(self):
        """Save annotation regions to JSON file"""
        os.makedirs(os.path.dirname(self.save_json) or ".", exist_ok=True)
        with open(self.save_json, "w", encoding="utf-8") as f:
            json.dump(self.regions, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 🎉 Ground truth file saved successfully to: {self.save_json}")

    def run(self):
        """Start the interactive annotation interface"""
        plt.show()

# =========================================================
# Feature map construction and evaluation module (multi-class upgraded)
# =========================================================
def build_instance_boundary_map(regions: List[Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    """Create binary boundary map from all instance contours"""
    h, w = shape[:2]
    boundary_map = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        contour = ensure_int_contour(r.get("contour", []))
        if contour.shape[0] < 3: continue
        cv2.polylines(boundary_map, [contour], isClosed=True, color=1, thickness=1)
    return boundary_map


def build_multiclass_label_map(regions: List[Dict[str, Any]], shape: Tuple[int, int],
                               label_mapping: Dict[str, int] = None) -> np.ndarray:
    """Create multi-class semantic segmentation label map"""
    h, w = shape[:2]
    label_map = np.zeros((h, w), dtype=np.int32)
    for idx, r in enumerate(regions):
        contour = ensure_int_contour(r.get("contour", []))
        if contour.shape[0] < 3: continue

        semantic_label = str(r.get("matched_legend_id", r.get("label", "")))
        if label_mapping and semantic_label in label_mapping:
            cls_id = label_mapping[semantic_label]
        else:
            cls_id = idx + 1  # Default to instance ID mode

        cv2.drawContours(label_map, [contour], -1, cls_id, thickness=-1)
    return label_map


def binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate binary segmentation metrics (IoU, Dice, Precision, Recall, F1)"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    pa = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"IoU": float(iou), "Dice": float(dice), "PixelAccuracy": float(pa), "Precision": float(precision),
            "Recall": float(recall), "F1": float(f1)}


def boundary_f1_score_v2(pred_regions: List[Dict], gt_regions: List[Dict], shape: Tuple[int, int],
                         tolerance: int = 10) -> Dict[str, float]:
    """Calculate boundary F1 score with pixel tolerance"""
    pred_b = build_instance_boundary_map(pred_regions, shape)
    gt_b = build_instance_boundary_map(gt_regions, shape)

    gt_dist = distance_transform_edt(1 - gt_b)
    pred_dist = distance_transform_edt(1 - pred_b)

    pred_match = ((pred_b > 0) & (gt_dist <= tolerance)).sum()
    gt_match = ((gt_b > 0) & (pred_dist <= tolerance)).sum()

    pred_total = max((pred_b > 0).sum(), 1)
    gt_total = max((gt_b > 0).sum(), 1)

    precision = pred_match / pred_total
    recall = gt_match / gt_total
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"BoundaryPrecision": float(precision), "BoundaryRecall": float(recall), "BoundaryF1": float(f1),
            "TolerancePx": int(tolerance)}


def average_boundary_distance(pred_regions: List[Dict], gt_regions: List[Dict], shape: Tuple[int, int]) -> Dict[
    str, float]:
    """Calculate average distance between predicted and ground truth boundaries"""
    pred_b = build_instance_boundary_map(pred_regions, shape)
    gt_b = build_instance_boundary_map(gt_regions, shape)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return {"MeanBoundaryDistance": float("inf")}

    gt_dist = distance_transform_edt(1 - gt_b)
    pred_dist = distance_transform_edt(1 - pred_b)

    d1 = gt_dist[pred_b > 0]
    d2 = pred_dist[gt_b > 0]
    return {"MeanBoundaryDistance": float((d1.mean() + d2.mean()) / 2.0)}


def multiclass_iou(pred: np.ndarray, gt: np.ndarray, ignore_bg=True) -> Dict[str, Any]:
    """Calculate mean IoU and per-class IoU for multi-class segmentation"""
    classes = sorted(set(np.unique(pred)).union(set(np.unique(gt))))
    if ignore_bg and 0 in classes:
        classes.remove(0)

    per_class = {}
    ious = []
    for c in classes:
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        iou = inter / max(union, 1)
        per_class[int(c)] = float(iou)
        ious.append(iou)

    return {"mIoU": float(np.mean(ious)) if len(ious) > 0 else 0.0, "per_class_IoU": per_class}


def evaluate_legend_accuracy(pred_regions: List[Dict], gt_regions: List[Dict], shape: Tuple[int, int],
                             iou_thresh: float = 0.5) -> Dict[str, float]:
    """
    Calculate legend/category matching accuracy through instance-level IoU matching. (Memory optimized version)
    """
    h, w = shape[:2]
    num_gt = len(gt_regions)
    num_pred = len(pred_regions)

    if num_gt == 0:
        return {"total_gt": 0, "matched_loc": 0, "correct_class": 0, "accuracy_given_loc": 0.0, "accuracy_overall": 0.0}

    def get_mask(region):
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt = ensure_int_contour(region.get("contour", []))
        if cnt.shape[0] >= 3:
            cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        return mask

    def get_bbox(region):
        """Calculate bounding rectangle of polygon [xmin, ymin, xmax, ymax]"""
        cnt = ensure_int_contour(region.get("contour", []))
        if cnt.shape[0] < 3: return (0, 0, 0, 0)
        x, y, bw, bh = cv2.boundingRect(cnt)
        return (x, y, x + bw, y + bh)

    # 1. Precompute BBox for all predictions (minimal memory usage)
    pred_bboxes = [get_bbox(r) for r in pred_regions]

    # 2. Calculate IoU pair by pair, avoid storing all masks in list
    iou_matrix = np.zeros((num_gt, max(num_pred, 1)))

    for i, gt_r in enumerate(gt_regions):
        if num_gt > 0:
            import sys
            percent = (i + 1) / num_gt * 100
            sys.stdout.write(f"\r   -> Performing pixel-level Mask IoU comparison: {percent:.1f}% [{i + 1}/{num_gt}] ")
            sys.stdout.flush()
            if i + 1 == num_gt:
                sys.stdout.write("\n")  # New line after completion

        gt_bbox = get_bbox(gt_r)
        if gt_bbox == (0, 0, 0, 0): continue

        gt_m = None  # Lazy initialization of GT mask

        for j, pr_r in enumerate(pred_regions):
            pr_bbox = pred_bboxes[j]
            if pr_bbox == (0, 0, 0, 0): continue

            # 3. Quick BBox intersection check: skip if no overlap
            if not (gt_bbox[2] < pr_bbox[0] or gt_bbox[0] > pr_bbox[2] or
                    gt_bbox[3] < pr_bbox[1] or gt_bbox[1] > pr_bbox[3]):

                # Generate mask only when BBoxes intersect
                if gt_m is None:
                    gt_m = get_mask(gt_r)
                    if gt_m.sum() == 0: break

                pr_m = get_mask(pr_r)
                inter = np.logical_and(gt_m, pr_m).sum()
                if inter > 0:
                    union = np.logical_or(gt_m, pr_m).sum()
                    iou_matrix[i, j] = inter / union

    matched_gt = 0
    correct_class = 0

    # 4. Greedy matching: find best IoU pred for each GT
    for i in range(num_gt):
        if num_pred == 0: break

        max_iou_idx = np.argmax(iou_matrix[i])
        max_iou = iou_matrix[i, max_iou_idx]

        # 5. Spatial match successful if IoU >= threshold
        if max_iou >= iou_thresh:
            matched_gt += 1

            gt_label = str(gt_regions[i].get("matched_legend_id", gt_regions[i].get("label", ""))).strip()
            pred_label = str(
                pred_regions[max_iou_idx].get("matched_legend_id", pred_regions[max_iou_idx].get("label", ""))).strip()

            if gt_label == pred_label:
                correct_class += 1

    # 6. Calculate metrics
    acc_given_match = correct_class / matched_gt if matched_gt > 0 else 0.0
    acc_overall = correct_class / num_gt if num_gt > 0 else 0.0

    return {
        "total_gt": int(num_gt),
        "matched_loc": int(matched_gt),
        "correct_class": int(correct_class),
        "accuracy_given_loc": float(acc_given_match),
        "accuracy_overall": float(acc_overall)
    }


def save_label_map_vis(label_map: np.ndarray, out_path: str):
    """Save color-visualized label map image"""
    h, w = label_map.shape[:2]
    palette = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 128, 0], [128, 0, 255], [0, 128, 255], [128, 255, 0], [255, 0, 128], [0, 255, 128],
        [128, 128, 255], [255, 128, 128], [128, 255, 128], [128, 128, 128], [200, 100, 50],
        [50, 200, 100], [100, 50, 200], [220, 180, 60],
    ], dtype=np.uint8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    unique_vals = np.unique(label_map)
    for v in unique_vals:
        if v < 0: continue
        vis[label_map == v] = palette[int(v) % len(palette)]
    save_rgb(out_path, vis)


def evaluate_prediction(img_path: str, pred_regions_json: str, gt_regions_json: str, out_dir: str, tolerance: int = 10):
    """Comprehensive evaluation of segmentation predictions against ground truth"""
    os.makedirs(out_dir, exist_ok=True)
    img = load_rgb(img_path)
    h, w = img.shape[:2]

    with open(pred_regions_json, "r", encoding="utf-8") as f: pred_regions = json.load(f)
    with open(gt_regions_json, "r", encoding="utf-8") as f: gt_regions = json.load(f)

    pred_map = build_multiclass_label_map(pred_regions, (h, w))
    gt_map = build_multiclass_label_map(gt_regions, (h, w))

    save_label_map_vis(pred_map, os.path.join(out_dir, "pred_label_map.png"))
    save_label_map_vis(gt_map, os.path.join(out_dir, "gt_label_map.png"))

    pred_fg = (pred_map > 0).astype(np.uint8)
    gt_fg = (gt_map > 0).astype(np.uint8)

    metrics_bin = binary_metrics(pred_fg, gt_fg)
    metrics_boundary = boundary_f1_score_v2(pred_regions, gt_regions, (h, w), tolerance=tolerance)
    metrics_dist = average_boundary_distance(pred_regions, gt_regions, (h, w))
    metrics_multi = multiclass_iou(pred_map, gt_map, ignore_bg=True)

    # New: Calculate instance-level legend matching accuracy
    metrics_legend = evaluate_legend_accuracy(pred_regions, gt_regions, (h, w), iou_thresh=0.5)

    # Save overlay image (instance boundaries only)
    vis = img.copy()
    pred_b = build_instance_boundary_map(pred_regions, (h, w))
    gt_b = build_instance_boundary_map(gt_regions, (h, w))
    overlap = (pred_b > 0) & (gt_b > 0)
    vis[(gt_b > 0) & (~overlap)] = [0, 255, 0]
    vis[(pred_b > 0) & (~overlap)] = [255, 0, 0]
    vis[overlap] = [255, 255, 0]
    save_rgb(os.path.join(out_dir, "boundary_overlay.png"), vis)

    # Modified: Add legend matching metrics to final output
    result = {
        "binary_metrics": metrics_bin,
        "boundary_metrics": metrics_boundary,
        "distance_metrics": metrics_dist,
        "multiclass_metrics": metrics_multi,
        "legend_accuracy_metrics": metrics_legend,
        "pred_regions_json": pred_regions_json,
        "gt_regions_json": gt_regions_json,
    }

    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved evaluation to: {out_dir}")

    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved evaluation to: {out_dir}")

    return result  # Return results for external usage


# =========================================================
# Command Line Interface
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # annotate/review mode
    p_anno = subparsers.add_parser("annotate", help="Interactive Review and Annotation")
    p_anno.add_argument("--img", required=True, help="Input image path")
    p_anno.add_argument("--save_json", required=True, help="Output Ground Truth JSON path")
    p_anno.add_argument("--pred_json", default=None, help="Optional: Load predictions as baseline")

    # evaluation mode
    p_eval = subparsers.add_parser("evaluate", help="Evaluate segmentation result")
    p_eval.add_argument("--img", required=True, help="Input image path")
    p_eval.add_argument("--pred_json", required=True, help="Prediction JSON file path")
    p_eval.add_argument("--gt_json", required=True, help="Ground Truth JSON file path")
    p_eval.add_argument("--out_dir", required=True, help="Output directory for evaluation results")
    p_eval.add_argument("--tolerance", type=int, default=10, help="Boundary tolerance in pixels")

    args = parser.parse_args()

    if args.mode == "annotate":
        annotator = ReviewAnnotator(
            img_path=args.img,
            save_json=args.save_json,
            pred_json=args.pred_json
        )
        annotator.run()

    elif args.mode == "evaluate":
        evaluate_prediction(args.img, args.pred_json, args.gt_json, args.out_dir, args.tolerance)


if __name__ == "__main__":
    main()