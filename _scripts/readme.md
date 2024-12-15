change working directory:
cd objectdet-yolov5-master

run training and export scripts with:
python _scripts/run_train.py --img 320 --batch 130 --epochs 300 --data configs/dataset.yaml --cache --device 0 --patience 15
python _scripts/run_export.py --img 320 --weights runs/train/exp/weights/best.pt --include tflite
