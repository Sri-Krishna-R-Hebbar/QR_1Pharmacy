import cv2
import os

def draw_boxes_on_image(image_path, boxes, out_path=None, color=(0,255,0), thickness=2):
    img = cv2.imread(image_path)
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        cv2.imwrite(out_path, img)
    return img
