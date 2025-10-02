# QR_1Pharmacy

This repository implements a YOLOv8-based pipeline for detecting QR codes in images.

It includes training, inference, and evaluation scripts, along with utilities for visualization.

---

## 📂 Project Structure

The project follows a standard structure for machine learning tasks:

    QR_1PHARMACY/

    ├── data/

    │ └── demo/            # Example dummy dataset (few samples for demo)

    ├── outputs/           # Training & evaluation results

    │ └── vis/yolov8_qr/

    │     ├── results.png

    │     ├── confusion_matrix.png

    │     └── submission_detection.json

    ├── src/               # Source code

    │ ├── datasets/        # Place dataset here (YOLO format) like the in the folder /data/demo

    │ ├── models/

    │ └── utils/

    ├── train.py           # Training script

    ├── infer.py           # Inference script

    ├── evaluate.py        # Evaluation script

    ├── data.yml           # Dataset config for YOLO

    ├── requirements.txt

    └── README.md


## 🚀 Getting Started

### 1. Clone Repo

```bash
git clone https://github.com/Sri-Krishna-R-Hebbar/QR_1Pharmacy.git
cd QR_1PHARMACY
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

Organize your dataset in the standard YOLO format under the `src/datasets/` folder:

```
datasets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
```

For demo purposes, check the sample data in `/data/demo/`.

**Update `data.yml`:**

Ensure your `data.yml` points to the correct paths and defines the classes:

**YAML**

```
train: ./src/datasets/train/images
val: ./src/datasets/test/images
nc: 1
names: ['qrcode']
```

### 5. Train Model

Run the training script. Results will be saved in the `outputs/` directory.

```bash
python train.py --data data.yml --model yolov8n.pt --epochs 60 --imgsz 640 --batch 16
```

### 6. Evaluate (optional)

This prints mAP/precision/recall on the `val` split set in `data.yml`

```bash
python evaluate.py --weights outputs/yolov8_qr/weights/best.pt --data data.yml
```

### 7. Inference

Run inference on new images using the trained weights.

```bash
python infer.py --weights outputs/yolov8_qr/weights/best.pt --source src/datasets/test/images --output outputs/submission_1.json --save-vis outputs/vis
```

### 8. Conversion of Submission in the Submission format (optional)

Converts the JSON output from infer.py into required submission format (change the paths as needed in src/utils/convert_submission.py)

```bash
python src/utils/convert_submission.py
```


## 📊 Results

Key results are saved in `outputs/vis/yolov8_qr/`:

* **Training curves:** `outputs/yolov8_qr/results.png`
* **Confusion matrix:** `outputs/yolov8_qr/confusion_matrix.png`
* **Final submission JSON (for Task 1):** `outputs/submission_detection1.json`

## 📦 Submission

The final submission JSON file for evaluation is included in this repository structure:

`outputs/submission_detection1.json`
