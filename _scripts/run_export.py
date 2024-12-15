import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 model export")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained weights (e.g., runs/train/exp/weights/best.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for export (default: cuda:0)")
    parser.add_argument("--include", type=str, default="tflite", help="Format to export (default: tflite)")

    args = parser.parse_args()

    # Construct the YOLOv5 export command
    command = [
        "python", "yolov5/export.py",
        "--weights", args.weights,
        "--device", args.device,
        "--include", args.include
    ]

    # Execute the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()