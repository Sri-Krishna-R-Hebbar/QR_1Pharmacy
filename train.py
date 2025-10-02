import argparse
from ultralytics import YOLO
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on QR dataset")
    p.add_argument('--data', default='data.yml', help='path to data yaml (train/val paths)')
    p.add_argument('--model', default='yolov8n.pt', help='pretrained YOLOv8 weights to start from')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', default='outputs', help='project dir to save runs')
    p.add_argument('--name', default='yolov8_qr', help='experiment name')
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Starting training with data={args.data}, model={args.model}, epochs={args.epochs}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name
    )
    print("Training finished. Best weights are saved in outputs/{}/weights/".format(args.name))

if __name__ == "__main__":
    main()
