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

    args = parser.parse_args()

    # Construct the YOLOv5 training command
    command = [
        "python", "yolov5/train.py",
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", args.data,
        "--weights", args.weights,
        "--device", args.device
    ]

    # Execute the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()