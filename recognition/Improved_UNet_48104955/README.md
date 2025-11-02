# 🧠 Improved U-Net for 2D OASIS Brain MRI Segmentation  
**Author:** Jose Leoncio
**Student ID:** 48104955  
**Course:** COMP3710 – Pattern Analysis (2025)  
**Model Folder:** `recognition/Improved_UNet_48104955/`  
**Architecture:** Improved U-Net (Pre-activation Residual Blocks + InstanceNorm)  
**Dataset:** OASIS (2D slices segmented into background, CSF, GM, WM)

---

## 📘 1. Project Overview

This project implements a **2D semantic segmentation model** for classifying brain MRI slices from the OASIS dataset into four tissue types: background, cerebrospinal fluid (CSF), gray matter (GM), and white matter (WM).  

The model used is an **Improved U-Net**, which enhances the standard U-Net architecture with:
- **Residual skip connections**
- **Pre-activation blocks**
- **Instance Normalization**

The objective, as specified by the COMP3710 recognition assignment, is to achieve a **Dice Similarity Coefficient (DSC) ≥ 0.90** for each class on the held-out test set.

---

## 🧠 2. Dataset and Preprocessing

The dataset is organized into **train**, **val**, and **test** splits.


### Processing Steps
- Images are grayscale MRIs normalized to `[0,1]`.
- Masks are remapped to integer classes `{0,1,2,3}`.
- If mask colors are grayscale intensities (e.g. `{0,85,170,255}`), the loader automatically converts them.

| Label | Class | Description |
|-------|--------|-------------|
| 0 | Background | Non-brain area |
| 1 | CSF | Cerebrospinal fluid |
| 2 | GM | Gray matter |
| 3 | WM | White matter |

---

## ⚙️ 3. Model Architecture

### Improved U-Net
- **Encoder:** Downsampling convolutions with residual blocks.  
- **Bottleneck:** Two pre-activation residual blocks.  
- **Decoder:** Upsampling with skip connections.  
- **Normalization:** InstanceNorm2d for better contrast invariance.  
- **Activation:** LeakyReLU (α=0.01).  
- **Output:** 1×1 convolution → 4 channels (per-class logits).

This structure maintains U-Net’s spatial precision while improving feature reuse and stability.

---

## 🧩 4. Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-5 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 8 |
| Epochs | 20 |
| Gradient Clip | 1.0 |
| Loss Function | 0.7 × Dice + 0.3 × CrossEntropy |

Model checkpoints are automatically saved as `best_unet.pth` whenever the validation mean Dice improves.

---

## 🚀 5. How to Run

### 1️⃣ Environment Setup
```bash
conda create -n improved_unet python=3.10 -y
conda activate improved_unet
conda install -c conda-forge pytorch torchvision pillow numpy -y