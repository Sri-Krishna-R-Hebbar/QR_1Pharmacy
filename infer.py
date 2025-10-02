import argparse
import os
import sys
import cv2
from ultralytics import YOLO
from src.utils.io import list_images, convert_preds_to_submission
from src.utils.visualize import draw_boxes_on_image

def run_inference(weights, source_folder, output_json='outputs/submission_detection_1.json',
                  conf=0.25, imgsz=640, save_crops=None, save_vis_dir=None):
    model = YOLO(weights)
    imgs = list_images(source_folder)
    preds_results = []
    print(f"Running inference on {len(imgs)} images...")

    for img_name in imgs:
        img_path = os.path.join(source_folder, img_name)
        # run prediction for this single image (returns list of Results objects)
        results = model.predict(source=img_path, conf=conf, imgsz=imgsz)
        # results is list-like; we take first
        r = results[0]
        boxes_np = []
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            boxes_np = r.boxes.xyxy.cpu().numpy().tolist()
        else:
            boxes_np = []
        preds_results.append({'image_path': img_path, 'boxes': boxes_np})

        # optional: save visualization with boxes
        if save_vis_dir:
            vis_path = os.path.join(save_vis_dir, os.path.splitext(img_name)[0] + ".jpg")
            if boxes_np:
                draw_boxes_on_image(img_path, boxes_np, out_path=vis_path)

    # Build submission JSON (no decoding for stage1)
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    convert_preds_to_submission(preds_results, output_json, save_crops_dir=save_crops)
    print(f"Wrote submission -> {output_json}")
    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='path to weights (.pt)')
    parser.add_argument('--source', required=True, help='folder with test images (e.g., src/datasets/test/images)')
    parser.add_argument('--output', default='outputs/submission_detection_1.json', help='output JSON path')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--save-crops', default=None, help='optional dir to save crops')
    parser.add_argument('--save-vis', default=None, help='optional dir to save visualization images with boxes')
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source_folder=args.source,
        output_json=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        save_crops=args.save_crops,
        save_vis_dir=args.save_vis
    )
