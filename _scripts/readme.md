run run_train.py with:

python scripts/run_train.py --img 320 --batch 130 --epochs 300 --data configs/dataset.yaml --cache --device 0 --patience 15




run run_export.py with:

python scripts/run_export.py --img 320 --weights runs/train/exp/weights/best.pt --include tflite