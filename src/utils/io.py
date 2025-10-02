import os
import json
from PIL import Image
import cv2
import numpy as np

def list_images(folder, exts=('.jpg','.jpeg','.png')):
    imgs = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return imgs

def yolo_txt_to_xyxy(line, img_w, img_h):
    # YOLO format: class x_center y_center width height (all normalized)
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, x_c, y_c, w, h = map(float, parts[:5])
    x1 = int((x_c - w/2) * img_w)
    y1 = int((y_c - h/2) * img_h)
    x2 = int((x_c + w/2) * img_w)
    y2 = int((y_c + h/2) * img_h)
    # clamp
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = max(0, min(img_w-1, x2)); y2 = max(0, min(img_h-1, y2))
    return [x1, y1, x2, y2]

def load_yolo_labels_for_image(label_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            xyxy = yolo_txt_to_xyxy(line, img_w, img_h)
            if xyxy:
                bboxes.append(xyxy)
    return bboxes

def convert_preds_to_submission(preds_results, out_json_path, save_crops_dir=None, decode_callback=None):
    """
    preds_results: list of dicts, each dict for an image containing:
       - 'image_path': path
       - 'boxes': list of [x1,y1,x2,y2]
    decode_callback (optional): function(crop_image_np) -> decoded_string or None
    """
    submission = []
    os.makedirs(os.path.dirname(out_json_path) or '.', exist_ok=True)
    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)

    for r in preds_results:
        img_path = r['image_path']
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        qrs = []
        for idx, box in enumerate(r['boxes']):
            x1,y1,x2,y2 = map(int, box)
            entry = {"bbox":[x1,y1,x2,y2]}
            if decode_callback:
                padx = int(0.08 * (x2-x1)); pady = int(0.08 * (y2-y1))
                x1p = max(0, x1-padx); y1p = max(0, y1-pady)
                x2p = min(w-1, x2+padx); y2p = min(h-1, y2+pady)
                crop = img[y1p:y2p, x1p:x2p]
                val = decode_callback(crop)
                if val:
                    entry["value"] = val
            qrs.append(entry)
            # optionally save crop
            if save_crops_dir:
                crop_file = os.path.join(save_crops_dir, f"{img_id}_{idx}.png")
                cv2.imwrite(crop_file, img[y1:y2, x1:x2])
        submission.append({"image_id": img_id, "qrs": qrs})

    with open(out_json_path, 'w') as f:
        json.dump(submission, f, indent=2)
    return submission
