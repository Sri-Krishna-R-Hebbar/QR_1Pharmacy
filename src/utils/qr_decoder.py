import cv2
from pyzbar.pyzbar import decode as zbar_decode


def crop_and_decode(image, bbox):
    """
    Crop region from bbox and decode QR code.
    bbox = [x_min, y_min, x_max, y_max]
    Returns decoded string if found, else None.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    crop = image[y_min:y_max, x_min:x_max]

    detector = cv2.QRCodeDetector()
    val, pts, _ = detector.detectAndDecode(crop)

    if val:
        return val.strip()

    decoded_objs = zbar_decode(crop)
    if decoded_objs:
        return decoded_objs[0].data.decode("utf-8").strip()

    return None
