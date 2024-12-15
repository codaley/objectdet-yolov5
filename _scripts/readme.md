TERMINAL TAB 1
change working directory:
cd objectdet-yolov5-master

launch tensorboard session:
pip install "tensorflow-cpu<=2.15.1"
tensorboard --logdir=runs/train --host=localhost --port=6006


TERMINAL TAB 2
change working directory:
cd objectdet-yolov5-master

call training script with:
python _scripts/run_train.py --img 320 --batch 130 --epochs 170 --data _configs/dataset.yaml --cache --device 0 --patience 0

call export script with:
python _scripts/run_export.py --img 320 --weights runs/train/exp12/weights/best.pt --include tflite
