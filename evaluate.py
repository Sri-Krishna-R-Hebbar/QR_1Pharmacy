import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='path to weights .pt')
    parser.add_argument('--data', default='data.yml', help='path to data yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    print(f"Running evaluation on data: {args.data}")
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("Evaluation complete.")
    print(metrics)

if __name__ == "__main__":
    main()
