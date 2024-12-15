import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 training with custom flags")
    parser.add_argument("--img", type=int, default=320, help="Image size (default: 320)")
    parser.add_argument("--batch", type=int, default=130, help="Batch size (default: 130)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs (default: 300)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--weights", type=str, default="yolov5m.pt", help="Initial weights (default: yolov5m.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device (default: cuda:0)")
    
    # For --cache, we use 'action="store_true"' so if the user includes --cache, it sets args.cache = True.
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")

    # For --patience, we assume it takes an integer. Set a sensible default or leave it optional.
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience (default: 100)")

    args = parser.parse_args()

    # Construct the YOLOv5 training command
    command = [
        "python", "objectdet-yolov5-master/train.py",
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", args.data,
        "--weights", args.weights,
        "--device", args.device
    ]

    if args.cache:
        command.append("--cache")

    if args.patience is not None:
        command.extend(["--patience", str(args.patience)])

    # Execute the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
