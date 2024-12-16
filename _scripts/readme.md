# TERMINAL TAB 1
### Create new conda environment with Python 3.9 and activate it

### Install project requirements
pip install -r requirements.txt

### Install CUDA specific torch and torchvision packages:
pip uninstall torch torchvision -y ; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

### Check CUDA availability:
python -c "import torch; \
print('CUDA Available:', torch.cuda.is_available()); \
print('CUDA Device Count:', torch.cuda.device_count()); \
print('Current CUDA Device:', torch.cuda.current_device() if torch.cuda.is_available() else 'No GPU'); \
print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

### Call training script with:
python _scripts/run_train.py --img 320 --batch 130 --epochs 170 --data _configs/dataset.yaml --cache --device 0 --patience 0

### Once training completes call export script with:
python _scripts/run_export.py --img 320 --weights runs/train/exp12/weights/best.pt --include tflite


# TERMINAL TAB 2
### Launch TensorBoard session:
tensorboard --logdir=runs/train --host=localhost --port=6006