import argparse
import os
import cv2
import json
from ultralytics import YOLO
from src.utils.io import list_images
from src.utils.visualize import draw_boxes_on_image
from src.utils.qr_decoder import crop_and_decode


def run_inference(weights, source_folder, output_json='outputs/submission_detection_1.json',
                  conf=0.25, imgsz=640, save_crops=None, save_vis_dir=None, decode=False):
    model = YOLO(weights)
    imgs = list_images(source_folder)
    results_list = []

    print(f"Running inference on {len(imgs)} images...")

    for img_name in imgs:
        img_path = os.path.join(source_folder, img_name)
        image = cv2.imread(img_path)

        results = model.predict(source=img_path, conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]

        entry = {"image_id": img_name.split("_jpg")[0], "qrs": []}
        boxes_np = []

        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            boxes_np = r.boxes.xyxy.cpu().numpy().tolist()
        else:
            boxes_np = []

        for box in boxes_np:
            x_min, y_min, x_max, y_max = map(int, box)
            qr_entry = {"bbox": [x_min, y_min, x_max, y_max]}

            if decode:
                val = crop_and_decode(image, [x_min, y_min, x_max, y_max])
                qr_entry["value"] = val

            entry["qrs"].append(qr_entry)

        results_list.append(entry)

        if save_vis_dir:
            os.makedirs(save_vis_dir, exist_ok=True)
            vis_path = os.path.join(save_vis_dir, os.path.splitext(img_name)[0] + ".jpg")
            if boxes_np:
                draw_boxes_on_image(img_path, boxes_np, out_path=vis_path)

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=2)

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
    parser.add_argument('--decode', action='store_true', help='enable QR decoding (Stage 2 bonus)')
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source_folder=args.source,
        output_json=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        save_crops=args.save_crops,
        save_vis_dir=args.save_vis,
        decode=args.decode
    )
