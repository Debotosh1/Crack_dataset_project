# 🧠 Text-Conditioned Segmentation using SAM + CLIP

This project implements a **text-conditioned image segmentation model** using the **Segment Anything Model (SAM)** and a **CLIP text encoder**.

The model generates segmentation masks from natural language prompts such as:

- "segment crack"
- "segment wall crack"


---

# 🚀 Overview

We extend SAM to support **text prompts** by injecting CLIP text embeddings into the prompt space.

> 🔑 Instead of bounding boxes or points, the model uses **text as a prompt**.

---

# 🏗️ Architecture

- **Image Encoder**: SAM ViT-B *(frozen)*
- **Text Encoder**: CLIP ViT-B/32 *(frozen)*
- **Projection Layer**: 512 → 256 *(trainable)*
- **Mask Decoder**: SAM decoder *(trainable)*

---

# 📊 Model Statistics

| Component | Value |
|----------|------|
| Total Parameters | **157.03M** |
| Trainable Parameters | **4.20M** |
| Frozen Parameters | **152.84M** |

---

# 📂 Dataset

## 🔹 Datasets Used

### 1. Crack Dataset
Prompts:
- "segment crack"
- "segment wall crack"



---

## 📈 Dataset Split

| Split | Images |
|------|--------|
| Train | **7073** |
| Validation | **1397** |
| Test | **430** |

---

# 🧹 Data Preparation

### 🔹 Mask Generation
- COCO annotations → binary masks
- Polygon filling using OpenCV

### 🔹 Mask Format
- Single-channel PNG
- Resolution: **1024 × 1024**
- Pixel values: `{0, 255}`

---

# 🔄 Preprocessing

- Resize images to **1024 × 1024**
- Normalize pixel values to `[0,1]`
- Convert masks to binary format

---

# 🔧 Data Augmentation

### ✅ Applied
- Horizontal Flip  
- Vertical Flip  
- Rotation (±10°)  
- Brightness & Contrast  
- Gaussian Noise (low)  

 

---

# 🧮 Loss Function | Metric | Fail Case Analysis

We use a combination of Focal Loss and Dice Loss:

```python
Loss = 0.3 * FocalLoss + 0.7 * DiceLoss
📊 Results
==========

🔹 Test Performance
-------------------

| Metric | Value |
|--------|------|
| mIoU | **0.5782** |
| Dice Score | **0.7149** |

---

⚡ Inference Performance
-----------------------

| Metric | Value |
|--------|------|
| Avg Inference Time | **0.2112 sec/image** |
| FPS | **4.74** |

---

❌ Failure Cases
===============

1. Partial Crack Detection
--------------------------

- Model fails to detect the **entire crack**, especially thin regions  
- Weak response in low-contrast areas  



---

2. False Positives on Background
-------------------------------

- Model detects cracks in **background textures or noise**

