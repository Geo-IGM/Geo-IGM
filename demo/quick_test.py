import os
import sys
import time
import threading

# 1. Import core pipeline
try:
    from main_demo import run as run_pipeline
except ImportError:
    print("❌ Error: Cannot import main_demo.py, please check the filename.")
    sys.exit(1)

# 2. Force import evaluation module
try:
    # Note: Function name in eval_demo.py is evaluate_prediction
    from eval_demo import evaluate_prediction
except ImportError as e:
    print(f"❌ Error: Cannot import eval_demo.py! Please ensure the file exists and contains evaluate_prediction. (Detail: {e})")
    sys.exit(1)


class LoadingBar:
    """A background thread utility for dynamic terminal loading animation"""

    def __init__(self, desc="   -> Running spatial matching and metric calculation"):
        self.desc = desc
        self.done = False
        self.thread = threading.Thread(target=self.animate)

    def animate(self):
        bar_width = 20
        pos = 0
        direction = 1
        while not self.done:
            bar = [' '] * bar_width
            for i in range(5):
                if 0 <= pos + i < bar_width:
                    bar[pos + i] = '█'

            sys.stdout.write(f'\r{self.desc} (Please wait) [{"".join(bar)}] ')
            sys.stdout.flush()

            pos += direction
            if pos == bar_width - 5 or pos == 0:
                direction *= -1
            time.sleep(0.08)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done = True
        self.thread.join()
        sys.stdout.write(f'\r{self.desc} [{"█" * 20}] ✅ Done!     \n')
        sys.stdout.flush()

def main():
    print("========================================")
    print("🚀 Geological Map Parsing & Automatic Evaluation Test")
    print("========================================")

    # --- Path Configuration ---
    test_img_path = r"D:\Desktop\IGM\data\maps\sample_cgs.jpg"
    save_dir = "output"
    eval_out_dir = os.path.join(save_dir, "eval_results")

    regions_json = os.path.join(save_dir, "regions_ui.json")
    gt_file_path = "gt_regions.json"

    if not os.path.exists(test_img_path):
        print(f"❌ Test image not found: {test_img_path}")
        return

    # --- Stage 1: Run core pipeline ---
    print(f"\n[Stage 1] Starting map parsing pipeline...")
    try:
        run_pipeline(test_img_path)
        print("✅ Parsing pipeline completed!\n")
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return

    # --- Stage 2: Run objective evaluation ---
    print(f"[Stage 2] Starting algorithm evaluation...")

    if not os.path.exists(regions_json):
        print(f"❌ Prediction file not found: {regions_json}, please check pipeline output.")
        return
    if not os.path.exists(gt_file_path):
        print(f"❌ Ground truth file not found: {gt_file_path}, please check path and annotation.")
        return

    metrics = None
    try:
        print(f"   -> Loading prediction: {regions_json}")
        print(f"   -> Loading ground truth: {gt_file_path}")

        metrics = evaluate_prediction(
            img_path=test_img_path,
            pred_regions_json=regions_json,
            gt_regions_json=gt_file_path,
            out_dir=eval_out_dir,
            tolerance=10
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # Show evaluation summary
    if metrics:
        iou = metrics['binary_metrics']['IoU']
        bf = metrics['boundary_metrics']['BoundaryF1']
        mbd = metrics['distance_metrics']['MeanBoundaryDistance']

        total_gt = metrics['legend_accuracy_metrics']['total_gt']
        matched_loc = metrics['legend_accuracy_metrics']['matched_loc']
        agl = metrics['legend_accuracy_metrics']['accuracy_given_loc']
        oa = metrics['legend_accuracy_metrics']['accuracy_overall']

        sr = matched_loc / total_gt if total_gt > 0 else 0.0

        print("\n📊 Final Core Evaluation Metrics (6 Dimensions):")
        print("-" * 60)
        print(f"   ➤ IoU ↑ (Binary IoU):                {iou * 100:.2f}%")
        print(f"   ➤ BF  ↑ (Boundary F1):               {bf * 100:.2f}%")

        if mbd == float('inf'):
            print(f"   ➤ MBD ↓ (Mean Boundary Distance):    Unmatched (Inf)")
        else:
            print(f"   ➤ MBD ↓ (Mean Boundary Distance):    {mbd:.2f} px")

        print(f"   ➤ SR  ↑ (Spatial Matching Rate):      {sr * 100:.2f}%  ({matched_loc}/{total_gt})")
        print(f"   ➤ AGL ↑ (Accuracy Given Location):    {agl * 100:.2f}%")
        print(f"   ➤ OA  ↑ (Overall Accuracy):           {oa * 100:.2f}%")
        print("-" * 60)
        print(f"📂 Detailed metrics and maps saved to: {eval_out_dir}")
        print("🎉 End-to-end automatic test completed successfully!")


if __name__ == "__main__":
    main()