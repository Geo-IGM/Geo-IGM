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
# Basic Utilities
# =========================================================
def load_rgb(img_path: str) -> np.ndarray:
    img = sio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def save_rgb(path: str, img_rgb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def ensure_int_contour(contour) -> np.ndarray:
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim != 2 or cnt.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.int32)
    return cnt.astype(np.int32)

# =========================================================
# Human-Machine Collaboration: Interactive Review Editor (Clean Version - Thin Black Highlight)
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
        self.show_all_labels = False  # Hide all labels by default

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
        print("\n" + "=" * 60)
        print("💡 交互式审核与真值修改器 (清爽版)")
        print("【全局快捷键】")
        print("  t : 开启/关闭全局类别标签显示 (默认隐藏)")
        print("  s : 保存当前结果")
        print("  q : 保存并退出")
        print("  n : 进入 [新增绘制] 模式")
        print("  esc: 退回 [审核编辑] 模式")
        print("\n【审核编辑模式 (默认)】")
        print("  鼠标悬停在区域内 : 自动高亮该区域(细黑线)，并显示顶点与类别")
        print("  左键按住顶点 : 拖拽修改多边形边界")
        print("  鼠标悬停 + d : 删除该区域")
        print("  鼠标悬停 + c : 修改该区域的类别 (终端输入)")
        print("\n【新增绘制模式 (按 n 键进入)】")
        print("  左键 : 添加顶点 | 右键 : 撤销上一个顶点 | 回车 : 完成")
        print("=" * 60 + "\n")

    def find_nearest_element(self, x, y) -> Tuple[int, Tuple[int, int]]:
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

            # --- Core Visual Optimization Logic ---
            if is_hovered:
                # On hover: thin black line, small black vertices, red text label
                self.ax.plot(xs_cl, ys_cl, color='black', linewidth=1.2)
                self.ax.scatter(xs, ys, color='black', s=12, alpha=0.8)

                cx, cy = np.mean(xs), np.mean(ys)
                label = str(r.get("matched_legend_id", r.get("label", r.get("id", "?"))))
                self.ax.text(cx, cy, label, color='white', fontsize=11, fontweight='bold',
                             ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))
            else:
                # Non-hover: thin green line, semi-transparent, no vertices
                self.ax.plot(xs_cl, ys_cl, color='lime', linewidth=1.0, alpha=0.6)

                # Show global labels if enabled
                if self.show_all_labels:
                    cx, cy = np.mean(xs), np.mean(ys)
                    label = str(r.get("matched_legend_id", r.get("label", r.get("id", "?"))))
                    self.ax.text(cx, cy, label, color='white', fontsize=8,
                                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

        # Highlight the hovered vertex (cyan to avoid confusion with black lines)
        if self.hovered_vertex and self.mode == 'edit':
            ri, pi = self.hovered_vertex
            px, py = self.regions[ri]["contour"][pi]
            self.ax.scatter([px], [py], color='cyan', s=80, edgecolors='white', zorder=5)

        # Draw the region being created
        if self.mode == 'draw' and len(self.current_points) > 0:
            pts = np.array(self.current_points)
            self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1)
            self.ax.scatter(pts[:, 0], pts[:, 1], c='red', s=5)# Size of points for new region

        self.fig.canvas.draw_idle()

    def on_move(self, event):
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
        if event.button == 1 and self.dragging_vertex:
            self.dragging_vertex = None
            self.redraw()

    def on_key(self, event):
        if event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)
        elif event.key == "t":
            self.show_all_labels = not self.show_all_labels
            print(f"[INFO] 全局标签显示已 {'开启' if self.show_all_labels else '关闭'}")
            self.redraw()
        elif event.key == "n":
            self.mode = 'draw'
            self.current_points = []
            print("[INFO] 切换至绘制模式 (左键加点，回车完成)")
            self.hovered_region_idx, self.hovered_vertex = -1, None
            self.redraw()
        elif event.key == "escape":
            self.mode = 'edit'
            self.current_points = []
            print("[INFO] 退出绘制，返回编辑模式")
            self.redraw()

        if self.mode == 'edit':
            if event.key == "d" and self.hovered_region_idx != -1:
                removed = self.regions.pop(self.hovered_region_idx)
                print(f"[INFO] 已删除区域: {removed.get('matched_legend_id', removed.get('id'))}")
                self.hovered_region_idx = -1
                self.redraw()

            elif event.key == "c" and self.hovered_region_idx != -1:
                r = self.regions[self.hovered_region_idx]
                old_cls = r.get('matched_legend_id', r.get('label', '?'))

                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()

                prompt_text = f"当前类别为 '{old_cls}'。\n请输入新类别名 (直接确认或取消则保持不变):"
                new_cls = simpledialog.askstring("修改类别", prompt_text, parent=root)
                root.destroy()

                if new_cls is not None and new_cls.strip():
                    r['matched_legend_id'] = new_cls.strip()
                    r['label'] = new_cls.strip()
                    r['reviewed'] = True
                    print(f"[INFO] 类别已更新为: {new_cls.strip()}")
                    self.save()

                self.redraw()

        elif self.mode == 'draw':
            if event.key == "enter" and len(self.current_points) >= 3:
                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()
                new_cls = simpledialog.askstring(
                    "输入类别",
                    "多边形绘制闭合。\n请输入该区域的类别名 (如 '砂岩'):",
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
                print(f"[INFO] 已添加新区域 id={self.next_id}, 类别={new_cls.strip()}")
                self.next_id += 1
                self.current_points = []
                self.mode = 'edit'
                self.redraw()
                self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.save_json) or ".", exist_ok=True)
        with open(self.save_json, "w", encoding="utf-8") as f:
            json.dump(self.regions, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 🎉 真值文件已成功保存至: {self.save_json}")

    def run(self):
        plt.show()

# =========================================================
# Feature Map & Evaluation Module (Multi-Class Upgrade)
# =========================================================
def build_instance_boundary_map(regions: List[Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    boundary_map = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        contour = ensure_int_contour(r.get("contour", []))
        if contour.shape[0] < 3: continue
        cv2.polylines(boundary_map, [contour], isClosed=True, color=1, thickness=1)
    return boundary_map


def build_multiclass_label_map(regions: List[Dict[str, Any]], shape: Tuple[int, int],
                               label_mapping: Dict[str, int] = None) -> np.ndarray:
    h, w = shape[:2]
    label_map = np.zeros((h, w), dtype=np.int32)
    for idx, r in enumerate(regions):
        contour = ensure_int_contour(r.get("contour", []))
        if contour.shape[0] < 3: continue

        semantic_label = str(r.get("matched_legend_id", r.get("label", "")))
        if label_mapping and semantic_label in label_mapping:
            cls_id = label_mapping[semantic_label]
        else:
            cls_id = idx + 1  # Default instance ID mode

        cv2.drawContours(label_map, [contour], -1, cls_id, thickness=-1)
    return label_map


def binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
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
    Calculate legend ID (class) matching accuracy via instance-level IoU matching. (Memory optimized version)
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
        """Compute bounding box of polygon [xmin, ymin, xmax, ymax]"""
        cnt = ensure_int_contour(region.get("contour", []))
        if cnt.shape[0] < 3: return (0, 0, 0, 0)
        x, y, bw, bh = cv2.boundingRect(cnt)
        return (x, y, x + bw, y + bh)

    # 1. Precompute all predicted BBoxes (very low memory usage)
    pred_bboxes = [get_bbox(r) for r in pred_regions]

    # 2. Compute IoU pair by pair to avoid storing all masks in a list
    iou_matrix = np.zeros((num_gt, max(num_pred, 1)))

    for i, gt_r in enumerate(gt_regions):
        gt_bbox = get_bbox(gt_r)
        if gt_bbox == (0, 0, 0, 0): continue

        gt_m = None  # Lazy initialization of GT mask

        for j, pr_r in enumerate(pred_regions):
            pr_bbox = pred_bboxes[j]
            if pr_bbox == (0, 0, 0, 0): continue

            # 3. Fast BBox intersection check: skip if no overlap
            if not (gt_bbox[2] < pr_bbox[0] or gt_bbox[0] > pr_bbox[2] or
                    gt_bbox[3] < pr_bbox[1] or gt_bbox[1] > pr_bbox[3]):

                # Only generate mask if BBoxes intersect
                if gt_m is None:
                    gt_m = get_mask(gt_r)
                    if gt_m.sum() == 0: break  # Skip if GT area is zero

                pr_m = get_mask(pr_r)  # Automatically garbage collected
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

        # 5. Spatial match success if IoU >= threshold
        if max_iou >= iou_thresh:
            matched_gt += 1

            gt_label = str(gt_regions[i].get("matched_legend_id", gt_regions[i].get("label", ""))).strip()
            pred_label = str(
                pred_regions[max_iou_idx].get("matched_legend_id", pred_regions[max_iou_idx].get("label", ""))).strip()

            if gt_label == pred_label:
                correct_class += 1

    # 6. Compute metrics
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

    # === New: Call instance-level legend matching accuracy ===
    metrics_legend = evaluate_legend_accuracy(pred_regions, gt_regions, (h, w), iou_thresh=0.5)

    # Save overlay (instance boundaries only)
    vis = img.copy()
    pred_b = build_instance_boundary_map(pred_regions, (h, w))
    gt_b = build_instance_boundary_map(gt_regions, (h, w))
    overlap = (pred_b > 0) & (gt_b > 0)
    vis[(gt_b > 0) & (~overlap)] = [0, 255, 0]
    vis[(pred_b > 0) & (~overlap)] = [255, 0, 0]
    vis[overlap] = [255, 255, 0]
    save_rgb(os.path.join(out_dir, "boundary_overlay.png"), vis)

    # === Modified: Add legend matching metrics to final output ===
    result = {
        "binary_metrics": metrics_bin,
        "boundary_metrics": metrics_boundary,
        "distance_metrics": metrics_dist,
        "multiclass_metrics": metrics_multi,
        "legend_accuracy_metrics": metrics_legend,  # New instance-level metrics
        "pred_regions_json": pred_regions_json,
        "gt_regions_json": gt_regions_json,
    }

    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved evaluation to: {out_dir}")
    # End of evaluate_prediction in eval_demo.py
    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved evaluation to: {out_dir}")

    return result  # <--- Add this line to return results to quick_test.py


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # annotate/review
    p_anno = subparsers.add_parser("annotate", help="Interactive Review and Annotation")
    p_anno.add_argument("--img", required=True, help="Input image path")
    p_anno.add_argument("--save_json", required=True, help="Output Ground Truth JSON path")
    p_anno.add_argument("--pred_json", default=None, help="[Optional] Load predictions as baseline")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate segmentation result")
    p_eval.add_argument("--img", required=True)
    p_eval.add_argument("--pred_json", required=True)
    p_eval.add_argument("--gt_json", required=True)
    p_eval.add_argument("--out_dir", required=True)
    p_eval.add_argument("--tolerance", type=int, default=10)

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