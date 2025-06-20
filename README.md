# 🧠 Brain Tumor Segmentation using YOLOv8 and SAM2

This project implements an end-to-end pipeline for brain tumor segmentation from MRI images using:
- **YOLOv8** for object detection (tumor localization)
- **SAM2 (Segment Anything Model v2 by Meta AI)** for pixel-level segmentation of tumor regions

> This work was completed as part of the ARCH Technologies AI Internship – Category B Task 01.

---

## 📂 Project Structure

brain-tumor-segmentation/
├── Model Training.ipynb # Google Colab notebook with all code
├── runs/ # YOLO training results (after training)
├── images/ # Sample input and output images
│ ├── input/ # MRI brain scan samples
│ ├── yolo_output/ # YOLO detection results
│ └── sam_output/ # SAM2 segmentation masks
├── data.yaml # Dataset config file for YOLO training
├── README.md # This file


---

## 🚀 How It Works

### 1. **YOLOv11 (Ultralytics) – Tumor Detection**
- YOLOv8 is trained on a custom brain tumor dataset with bounding box annotations.
- The model predicts bounding boxes around tumors in MRI images.
- Trained for **10 epochs**, image size **640**.

### 2. **SAM2 (Segment Anything v2) – Tumor Segmentation**
- The bounding box from YOLO is passed as a prompt to SAM2.
- SAM2 returns a fine-grained segmentation mask that outlines the tumor region.
- The mask is overlaid on the original image for visual output.

---

## 📊 Results

| Component         | Result                          |
|-------------------|----------------------------------|
| YOLO Training     | 10 epochs, image size 640        |
| Detection         | Accurate bounding boxes          |
| Segmentation      | Clean pixel-level tumor masks    |

> Qualitative results show excellent segmentation accuracy on MRI scans.

---

## 🖼️ Sample Outputs

| YOLO Detection                        | SAM2 Segmentation                    |
|--------------------------------------|--------------------------------------|
| ![YOLO Output](images/yolo_output/sample.jpg) | ![SAM Output](images/sam_output/sample_mask.jpg) |

---

## 📦 Requirements

- Python 3.8+
- [Google Colab](https://colab.research.google.com/) (used for training and inference)
- `ultralytics` package
- `transformers`, `opencv-python`, `matplotlib`, `numpy`

### Install YOLOv11 & Dependencies (Colab):

pip install ultralytics opencv-python matplotlib transformers
🧪 Run YOLO Training

from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # nano model for quick training
model.train(data="data.yaml", epochs=10, imgsz=640)
✂️ Run SAM2 Segmentation

from transformers import SamModel, SamProcessor
from PIL import Image
import torch

# Load model and processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Load image and bounding box
image = Image.open("images/input/mri.jpg")
input_boxes = [[x1, y1, x2, y2]]  # From YOLO output

# Prepare input for SAM
inputs = processor(image, input_boxes=[[input_boxes]], return_tensors="pt")
outputs = model(**inputs)
📁 Dataset Format
Follows YOLO format (with images/train, labels/train, etc.)

Each label file contains:

<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
📽️ Demo Video
📺 Watch the demo on YouTube
A short video showing YOLO detection, SAM2 segmentation, and final overlays.

🧠 Acknowledgements
Ultralytics YOLOv11

Meta AI’s SAM

ARCH Technologies Internship Team
