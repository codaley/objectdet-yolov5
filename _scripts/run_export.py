import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 model export")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained weights (e.g., runs/train/exp/weights/best.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for export (default: cuda:0)")
    parser.add_argument("--include", type=str, default="tflite", help="Format to export (default: tflite)")
    parser.add_argument("--img", type=int, default=320, help="Export image size (default: 320)")

    args = parser.parse_args()

    # Construct the YOLOv5 export command
    command = [
        "python", "export.py",
        "--weights", args.weights,
        "--device", args.device,
        "--include", args.include,
        "--img", str(args.img)
    ]

    # Execute the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
