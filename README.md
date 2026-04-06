# 2025 NCKU Image Processing

This repository contains coursework implementations for the **Image Processing** course at National Cheng Kung University (NCKU).

The repository includes two main assignments:

- **IP_hw1**: Fundamental Image Processing Techniques with GUI Implementation
- **IP_hw2**: Medical Image Segmentation (MRI-based Muscle Segmentation)

---


# 🔹 IP_hw1 — Fundamental Image Processing

##  Overview

This assignment focuses on implementing **classical image processing techniques** and integrating them into a **Graphical User Interface (GUI)**.

The goal is to understand how low-level image operations work and how they affect image quality.

---

##  Implemented Techniques

### 1. Noise Removal
- Average Filter
- Median Filter
- Fourier Transform-based filtering

-> Removes noise from input images  
-> Demonstrates spatial vs frequency domain filtering :contentReference[oaicite:0]{index=0}

---

### 2. Image Sharpening
- Sobel Operator (edge enhancement)
- Fourier Transform sharpening

-> Enhances edges and details  
-> Highlights high-frequency components :contentReference[oaicite:1]{index=1}

---

### 3. Gaussian Filter Design
- Custom 5×5 Gaussian kernel

-> Smooths image while preserving structure  
-> Demonstrates convolution-based filtering :contentReference[oaicite:2]{index=2}

---

### 4. Low-Pass Filter
- Frequency-domain filtering

-> Removes high-frequency noise  
-> Produces blurred/smoothed output :contentReference[oaicite:3]{index=3}

---

##  GUI Implementation

A GUI is developed to interactively demonstrate image processing results.

### Features:
- Load image
- Apply filters (Smooth, Sharpen, Gaussian, Low-pass)
- Display original vs processed images

The GUI is implemented using:

- Python
- OpenCV
- PyQt5
- Matplotlib :contentReference[oaicite:4]{index=4}

---


# 🔹 IP_hw2 — Medical Image Segmentation

##  Overview

This assignment focuses on **medical image segmentation** using MRI images of the forearm.

The objective is to segment different anatomical structures from MRI scans.

---

##  Segmentation Target

Each MRI image is segmented into **three classes**:

- CT (Carpal Tunnel)
- FT (Flexor Tendons)
- MN (Median Nerve) :contentReference[oaicite:5]{index=5}

---

##  Dataset Description

- 10 sets of MRI images
- Each set includes:
  - T1-weighted image
  - T2-weighted image
  - Ground truth segmentation mask :contentReference[oaicite:6]{index=6}

---

##  Methodology

### Input
- T1 MRI image
- T2 MRI image

### Processing Pipeline
```bach
Load Image → Preprocessing → Feature Extraction → Segmentation → Evaluation
```

---

##  Evaluation Metric

### Dice Coefficient

Measures similarity between predicted mask and ground truth:

- Value range: 0 ~ 1
- Higher = better overlap

---

##  GUI System

A GUI is implemented to visualize segmentation results.

### Features:
- Load MRI dataset
- Switch between T1 / T2 images
- Display segmentation results
- Overlay predicted mask vs ground truth
- Compute Dice score in real time :contentReference[oaicite:8]{index=8}

---



# ⚠️ Notes

- Large files (datasets, videos, results) are excluded
- Only source code and reports are included

---
