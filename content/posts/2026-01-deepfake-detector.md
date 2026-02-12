---
title: "Hybrid Deepfake Detection Using Spectral Analysis and Convolutional Neural Networks"
date: 2026-01-15
categories: ["research"]
tags: ["deepfake", "CNN", "FFT", "machine-learning", "transfer-learning"]
summary: "A hybrid system for detecting AI-generated images combining FFT spectral analysis, Random Forest, and Xception CNN."
math: true
ShowToc: true
TocOpen: true
---

> **Author:** Radu-Petruț Vătase — [UPB](https://tcsi.ro/) | January 2026

**Keywords:** Deepfake, Convolutional Neural Networks, Spectral Analysis, Transfer Learning

---

## Abstract

This paper proposes a hybrid system for detecting synthetically generated images (deepfakes), combining mathematical frequency-domain analysis (Fast Fourier Transform), classical Machine Learning (Random Forest), and advanced Deep Learning techniques (CNN Xception). The comparative study demonstrates the limitations of manually extracted feature-based approaches (54% accuracy) versus the robustness of convolutional neural networks trained in the cloud on a Tesla T4 GPU. Results after 6 complete training epochs show a validation accuracy of 75.2% and an AUC score of 0.829, confirming the efficiency of the Xception architecture for deepfake detection. The system integrates an Explainable AI module based on GPT-4o for semantic interpretation of spectral anomalies, accessible through a public web interface.

---

## 1. Introduction

In today's digital era, the integrity of visual information is a major challenge. The advancement of Generative Adversarial Networks (GANs) has enabled the creation of photorealistic forgeries, raising serious concerns in cybersecurity, journalism, and identity authentication. This work addresses automatic detection through complementary methods, providing a solution accessible to researchers and the general public.

## 2. Theoretical Foundation

Detection of forgeries is based on the hypothesis that synthetic generation introduces statistical anomalies invisible to the human eye, but detectable mathematically and through deep neural networks.

### 2.1 Frequency Domain Analysis (FFT)

The 2D Fourier Transform identifies periodic artifacts introduced by upsampling operations in GANs:

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x,y) \cdot e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$

![FFT Analysis Example](/images/deepfake-detector/fft_example.png)
*Figure 1: FFT Analysis Example*

**Limitations:** Modern generators (StyleGAN3, DALL-E 3, Midjourney v6) produce images with spectral distributions nearly identical to real ones, making FFT analysis insufficient as a primary method. This finding justified the transition to Deep Learning.

### 2.2 Xception CNN Architecture

Xception uses Depthwise Separable Convolutions for superior computational efficiency. The architecture pre-trained on ImageNet (1.4M images) is adapted through Transfer Learning on 100,000 deepfake/real images.

## 3. Methodology and Architecture

The system combines three complementary components:

1. **FFT (Spectral Analysis):** Provides visual explainability
2. **Random Forest:** Classical baseline (5 manually extracted features)
3. **CNN Xception:** Decisive component (Transfer Learning)

### 3.1 Dataset and Training Platform

**Data used:**
- Random Forest: 2,041 images (local training)
- CNN: 100,000 images from "140k Real and Fake Faces" [4]
- Split: 80% training (80,000) / 20% validation (20,000)

| | |
|:---:|:---:|
| ![Real face](/images/deepfake-detector/real/exemplu_real.jpg) | ![Fake face](/images/deepfake-detector/fake/exemplu_fake.jpg) |
| *Real image — authentic photograph* | *Deepfake image — AI generated* |

**Infrastructure:**
Due to local hardware limitations (RTX 3050 4GB with CUDA incompatibilities), training was conducted in Google Colab on a Tesla T4 accelerator (16GB VRAM), reducing time from 40+ hours (local CPU) to approximately 2 hours/epoch.

### 3.2 Transfer Learning Strategy

**Phase 1 (Base Frozen):**
- Learning Rate: 0.001
- Loss: Binary Crossentropy
- Optimizer: Adam
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Phase 2 (Fine-Tuning):** Unfreeze last 20 layers, LR=0.0001 (planned, not executed due to Colab session expiration)

## 4. Implementation and Interface

### 4.1 Streamlit Web Application

The system is publicly available at:
**[https://deepfake-detector-tcsivtstcsitcsi.streamlit.app/](https://deepfake-detector-tcsivtstcsitcsi.streamlit.app/)**

**Deployment:** Streamlit Cloud with Python 3.11, TensorFlow 2.15.1, CNN model (80MB) hosted on GitHub.

**Explainable AI Features:**
- FFT 2D spectrum and radial profile visualization
- Semantic interpretation via OpenAI GPT-4o
- CNN prediction with probability breakdown
- Comparative analysis between methods (FFT/RF/CNN)

### 4.2 Additional Analyses

![FFT 2D Spectrum](/images/deepfake-detector/fft_2d_spectrum.png)
*Figure 2: 2D FFT Spectrum — frequency component visualization*

| | |
|:---:|:---:|
| ![Color Histogram](/images/deepfake-detector/color_histogram.png) | ![Gradient Magnitude](/images/deepfake-detector/gradient_magnitude.png) |
| *Color Histogram — RGB channel distribution* | *Gradient Magnitude — detail map* |

| | |
|:---:|:---:|
| ![Noise Pattern](/images/deepfake-detector/noise_pattern.png) | ![EXIF Metadata](/images/deepfake-detector/exif_metadata.png) |
| *Noise Pattern — noise analysis* | *EXIF Metadata — camera information* |

## 5. Experimental Results

### 5.1 Performance Comparison

| Method | Accuracy | AUC | Recall | Notes |
|--------|----------|-----|--------|-------|
| Random Forest | 54.0% | — | 20% (Fake) | Severe bias towards Real |
| **CNN Ep. 6 (Best)** | **75.20%** | **0.8286** | **74.84%** | **Optimal after LR decay** |
| CNN Epoch 3 | 74.67% | 0.8273 | 78.90% | Initial peak |

### 5.2 Training Evolution

| Epoch | Train Acc | Val Acc | Val AUC | Val Recall | Status |
|-------|-----------|---------|---------|------------|--------|
| 1 | 66.65% | 73.52% | 0.8150 | 78.04% | Saved |
| 2 | 70.46% | 74.43% | 0.8231 | 79.46% | Saved |
| 3 | 70.45% | 74.67% | 0.8273 | 78.90% | Initial peak |
| 4 | 70.09% | 73.23% | 0.8248 | 83.37% | Val drop |
| 5 | 70.06% | 74.40% | 0.8279 | 80.73% | LR→0.0005 |
| **6** | **70.26%** | **75.20%** | **0.8286** | **74.84%** | **Best final** |

**Observations:**
- **Rapid convergence:** 73.5% accuracy in the first epoch
- **ReduceLROnPlateau effect:** LR decreased to 0.0005 in Epoch 5 → improved Val Accuracy to 75.2% (Epoch 6)
- **Time/epoch:** ~33-37 minutes on Tesla T4 (2017-2212s)
- **Infrastructure limitation:** Colab session expired in Epoch 9/10 (free tier)

### 5.3 Comparative Analysis

- **Random Forest:** Weak generalization (54%), insufficient manual feature engineering
- **CNN Xception:** Strong generalization (75.2%), automatic feature learning, high AUC (0.829)

## 6. Conclusions and Future Directions

### Achievements

1. **Complete functional system:** Public deployment on Streamlit Cloud
2. **Validated performance:** 75.2% accuracy, AUC 0.829 on 20,000 validation images
3. **Explainability:** GPT-4o integration for graph and value interpretation
4. **Methodological comparison:** Highlighting FFT and Random Forest limitations vs. CNN

### Limitations

**Infrastructure:**
- Incomplete training (6/20 planned epochs) due to free Colab session expiration
- Suboptimal final model — requires Phase 2 (fine-tuning) for 88-92% accuracy

![Colab session expired](/images/deepfake-detector/colab_epoch_limitation.png)
*Figure 3: Google Colab session expired during Epoch 9 — free tier infrastructure limitation*

**Methodological:**
- Detection limited to static images (not video)
- Unknown performance on sophisticated deepfakes (professionally post-processed)
- Dataset with 2020-2024 images — latest AI-generated images may be harder to distinguish

### Future Directions

- **Complete training:** Finish Phase 1 (epochs 7-10) + Phase 2 (fine-tuning)
- **Video analysis:** Extend to video deepfake detection (frame-by-frame + temporal consistency)
- **Ensemble methods:** Combine Xception + EfficientNet + Vision Transformer for improved efficiency

---

## References

1. Chollet, F., "Xception: Deep Learning with Depthwise Separable Convolutions", *CVPR*, 2017.
2. Durall, R., et al., "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions", *CVPR*, 2020.
3. Karras, T., et al., "Analyzing and Improving the Image Quality of StyleGAN", *CVPR*, 2020.
4. Kaggle, "140k Real and Fake Faces Dataset", 2023.
5. OpenAI, "GPT-4 Technical Report", 2024.
6. Streamlit Inc., "Streamlit Community Cloud Documentation", 2025.
