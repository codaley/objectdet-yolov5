# Object Detection with YOLOv5m for TensorFlow Lite Deployment

### Overview

This project demonstrates the training and deployment of a custom object detection model using **YOLOv5m**, fine-tuned for detecting rocks and bags. The model is designed for efficient real-time inference on edge devices and is integrated into an **Android app** running on a **Pixel 7a**. The goal is to enable precise and lightweight object detection while ensuring user flexibility to customize the pipeline and app for their own use cases.

While this project is not explicitly intended for autonomous driving systems, it tackles challenges related to object misidentification in autonomous systems, such as distinguishing between rocks and bags, which can significantly impact decision-making and safety.

**Example Logic:**
- **Detecting a rock**: This could prompt a vehicle to swerve, avoiding potential damage.
- **Detecting a bag**: This could help reduce unnecessary evasive actions, improving efficiency and safety.

![](media/rockbag-yolov5-android.gif)

This project builds upon a previous implementation trained with the high-level **TensorFlow Lite Model Maker** library. While TensorFlow Lite Model Maker provides a user-friendly interface for quick prototyping, this new project leverages the power and flexibility of **PyTorch** to create an entirely custom model, resulting in significant improvements in accuracy and adaptability. By transitioning to PyTorch and the YOLOv5 framework, this project enables precise control over the training process, model architecture, and hyperparameter tuning, unlocking advanced capabilities for object detection.

### Why PyTorch?
- **Flexibility**: PyTorch offers granular control over every aspect of the model, from custom architectures to advanced optimization techniques.
- **Performance**: The YOLOv5m model achieves higher accuracy and real-world reliability, benefiting from PyTorch’s robust ecosystem and active development.
- **Scalability**: This project bolsters the model with **6x more data** than the original implementation, covering diverse environments, lighting conditions, close-ups, and distant views.
- **Precision**: The model is fine-tuned using higher-resolution images and carefully optimized hyperparameters, resulting in superior accuracy metrics and real-world performance.

By incorporating a significantly larger dataset and optimizing the training pipeline, this model excels in detecting objects with improved mean Average Precision (mAP) and high precision and recall scores. These enhancements make the model highly reliable for real-world usage scenarios, even in challenging environments.

YOLOv5m, a mid-size variant of the YOLOv5 family, balances detection accuracy and inference speed. With an input image size of **320x320x3**, the model is optimized for deployment on devices with limited computational resources, achieving excellent results for bounding box predictions and object classification.

---

## Project Setup and Folder Structure

After downloading the repository, you will notice the folder structure includes custom folders prefixed with `_` for easy identification. Ensure you extract the files correctly to avoid nested folders (i.e., `Downloads/objectdet-yolov5-master/objectdet-yolov5-master`). The final structure should look like this:

```
Downloads/
└── objectdet-yolov5-master/
    ├── _configs/
    │   └── dataset.yaml
    ├── _dataset/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── test/
    │   │   ├── images/
    │   │   └── labels/
    │   └── valid/
    │       ├── images/
    │       └── labels/
    ├── _scripts/
    │   ├── run_train.py
    │   └── run_export.py
    └── other repo files...
```

---

## Quickstart Guide

### 1. Prepare the Dataset

1. **Preprocess the Images**  
   Crop the images to 1:1 squares at their original resolution. When calling the training script with `--img <size>`, images will be downsampled, and bounding box annotations will automatically scale to fit the chosen resolution.

2. **Annotate the Images**  
   Annotate your images using [LabelImg](https://github.com/heartexlabs/labelImg) and save them in YOLO format.

3. **Configure the Dataset**  
   - **Copy** your `train`, `test`, and `valid` folders into the `_dataset/` directory.  
   - **Edit** the `dataset.yaml` file in `_configs/` to match your dataset:  
      - `nc`: Number of classes.  
      - `names`: List of class names.  
      - `train` and `val`: Paths to your training and validation datasets.

### Example `_configs/dataset.yaml`:
```yaml
train: _dataset/train/images
val: _dataset/valid/images

nc: 2
names: ['rock', 'bag']
```

---

### 2. Set Up the Environment

1. Create a new Conda environment with Python 3.9:
   ```bash
   conda create -n yolov5-env python=3.9 -y
   conda activate yolov5-env
   ```

2. Install the project requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Install CUDA-specific PyTorch and torchvision packages:
   ```bash
   pip uninstall torch torchvision -y ; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. Verify CUDA availability:
   ```bash
   python -c "import torch;    print('CUDA Available:', torch.cuda.is_available());    print('CUDA Device Count:', torch.cuda.device_count());    print('Current CUDA Device:', torch.cuda.current_device() if torch.cuda.is_available() else 'No GPU');    print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
   ```

---

### 3. Train the Model

Run the following command to start training:
```bash
python _scripts/run_train.py --img 320 --batch 130 --epochs 170 --data _configs/dataset.yaml --cache --device 0 --patience 0
```
- Adjust the flags (e.g., `--img`, `--batch`, `--epochs`, `--patience`) as needed.

### 4. Monitor Training Metrics
Launch TensorBoard in a separate terminal session to visualize the training progress:
```bash
tensorboard --logdir=runs/train --host=localhost --port=6006
```

---

### 5. Export the Model

Once training completes, export the model to TensorFlow Lite format:
```bash
python _scripts/run_export.py --img 320 --weights runs/train/exp/weights/best.pt --include tflite
```
The exported model (`best-fp16.tflite`) is optimized for deployment.

---

## Android App Integration

The Android app is designed for seamless use with the pre-trained model, which is already integrated and ready to run. For those interested in customization, the following steps explain how to integrate a custom-trained model into the app.

### 1. Download the Android App Files
Download the app files from [this Dropbox link](https://www.dropbox.com/scl/fi/swv7q108dfq47frencldi/yolov5-example-app.zip?rlkey=c8lsvm4ub6im4yo6hu8dpyh5y&st=tpba5k7w&dl=0).

### 2. Download and Set Up Android Studio
Ensure that you have [Android Studio](https://developer.android.com/studio) installed on your system. Open the `/android` directory in Android Studio. This setup has been tested with:

- **Android Studio Bumblebee**
- **Gradle Plugin Version**: 3.5.0
- **Gradle Version**: 5.4.1
- **Gradle JDK**: 11.0.11

### 3. Replace the Model
Navigate to `android/app/src/main/assets/` and replace the existing `best-fp16.tflite` with your trained model. The filename **must exactly match** `best-fp16.tflite`.

### 4. Update Input Size
If your model uses an input size other than **320x320**, navigate to:
```
android/app/src/main/java/org/tensorflow/lite/examples/detection/tflite/DetectorFactory.java
```
Edit the `inputSize` tag to match the resolution your model was trained on.

### 5. Adjust Detection Threshold
To modify the detection confidence threshold, navigate to:
```
android/app/src/main/java/org/tensorflow/lite/examples/detection/MainActivity.java
```
Update the following line:
```java
MINIMUM_CONFIDENCE_TF_OD_API = 0.80f;
```
- **Default**: 80% confidence.
- **Customize**: Increase for fewer false positives or decrease to capture more objects.

### 6. Modify Classes
To add custom classes, edit the `customclasses.txt` file in the `assets` folder. Ensure each class is listed on a new line.

### 7. Connect Your Android Device
Connect your Android device to your computer using a USB cable. Ensure that USB debugging is enabled on the device. You can enable USB debugging by navigating to:

```
Settings > Developer Options > USB Debugging
```

### 8. Run the App
Once you have replaced the files, made the necessary modifications, and connected your Android device, you can run the app by clicking the **Run** button in Android Studio. The app will be deployed to the connected Android device and will use the integrated model for real-time detection.

---

## Model Performance Overview 📊

### Key Metrics
- **mAP@0.5**: ~0.982  
- **mAP@0.5:0.95**: >0.85  
- **Precision**: >0.98  
- **Recall**: ~0.97  

### Training Loss Analysis
- **Box Loss**: ~0.012  
- **Cls Loss**: ~5e-4.  
- **Obj Loss**: ~4.5e-3.

### Validation Loss Analysis
- **Box Loss**: <0.012  
- **Cls Loss**: <1e-3.  
- **Obj Loss**: <2.2e-3. 

Validation metrics confirm strong generalization with no overfitting.

### Learning Rate Schedules
The learning rate scheduler progressively reduces step size, balancing rapid initial convergence with fine-tuning in later epochs. The smooth decay ensures stable optimization, preventing overshooting and guiding the model to an optimal solution.

![](media/tensorboard-metrics.png)

---

## Privacy Notice

**Training Data Privacy:** The dataset used to train this model contains images annotated for object detection. For privacy reasons and to protect personally identifiable information (PII) or sensitive data, the training dataset will not be shared publicly. All dataset annotations and images were used solely for the purpose of this project.

---

## Future Work

Potential enhancements include:

1. **Expanding the Dataset**: Increase the dataset diversity to include various weather conditions such as rain, fog, snow, and low-light scenarios. This would improve the model’s robustness and ensure reliable performance across a wide range of real-world environments.

2. **Detecting Additional Hazards**: Extend the model to recognize and classify a broader range of objects and hazards commonly encountered on roads, such as potholes, construction zones, debris, or pedestrians. This would enhance the model’s applicability for more comprehensive object detection in autonomous or semi-autonomous systems.

3. **Enhancing Precision with LiDAR Integration**: By incorporating LiDAR data captured from iPhones equipped with LiDAR scanners (e.g., iPhone 12 Pro, iPhone 13 Pro, and newer), the model could achieve significantly higher precision in depth estimation and object detection. Leveraging depth maps and point cloud data generated by Apple's ARKit API, combined with paired RGB images, the model would gain enhanced spatial understanding and improved detection accuracy.

---

## Acknowledgments

This project builds on the official [YOLOv5 repository](https://github.com/ultralytics/yolov5). Android integration leverages TensorFlow Lite example app for object detection.
