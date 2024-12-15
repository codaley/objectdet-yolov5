######################### TERMINAL TAB 1 #########################
change working directory:
cd objectdet-yolov5-master

install project requirements:
pip install -r requirements.txt

call training script with:
python _scripts/run_train.py --img 320 --batch 130 --epochs 170 --data _configs/dataset.yaml --cache --device 0 --patience 0

call export script with:
python _scripts/run_export.py --img 320 --weights runs/train/exp12/weights/best.pt --include tflite


######################### TERMINAL TAB 2 #########################
change working directory:
cd objectdet-yolov5-master

launch tensorboard session:
tensorboard --logdir=runs/train --host=localhost --port=6006
