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

{{< lang-toggle >}}

{{< lang en >}}

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

{{< /lang >}}

{{< lang ro >}}

**Cuvinte cheie:** Deepfake, Rețele neuronale convoluționale, Analiză spectrală, Transfer Learning

---

## Rezumat

Prezenta lucrare propune un sistem hibrid pentru detectarea imaginilor generate sintetic (deepfake), combinând analiza matematică în domeniul frecvenței (Fast Fourier Transform), algoritmi de Machine Learning clasic (Random Forest) și tehnici avansate de Deep Learning (CNN Xception). Studiul comparativ demonstrează limitările abordărilor bazate pe trăsături extrase manual (acuratețe 54%) față de robustețea rețelelor neuronale convoluționale antrenate în cloud pe GPU Tesla T4. Rezultatele obținute după 6 epoci complete de antrenare arată o acuratețe de validare de 75.2% și un scor AUC de 0.829, confirmând eficiența arhitecturii Xception pentru detectarea deepfake-urilor. Sistemul integrează un modul de Explainable AI bazat pe GPT-4o pentru interpretarea semantică a anomaliilor spectrale, fiind accesibil printr-o interfață web publică.

---

## 1. Introducere

În era digitală actuală, integritatea informației vizuale reprezintă o provocare majoră. Avansul rețelelor generative antagoniste (GANs) a permis crearea de falsuri fotorealiste, ridicând probleme serioase în domeniul securității cibernetice, jurnalismului și autentificării identității. Această lucrare abordează problema detecției automate prin metode complementare, oferind o soluție accesibilă cercetătorilor și publicului larg.

## 2. Fundamentare teoretică

Detectarea falsurilor se bazează pe ipoteza că generarea sintetică introduce anomalii statistice invizibile ochiului uman, dar detectabile matematic și prin rețele neuronale profunde.

### 2.1 Analiza în domeniul frecvenței (FFT)

Transformata Fourier 2D identifică artefactele periodice introduse de operațiile de upsampling din GAN-uri:

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x,y) \cdot e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$

![Exemplu analiză FFT](/images/deepfake-detector/fft_example.png)
*Figura 1: Exemplu analiză FFT*

**Limitări:** Generatoarele moderne (StyleGAN3, DALL-E 3, Midjourney v6) produc imagini cu distribuții spectrale aproape identice cu cele reale, făcând analiza FFT insuficientă ca metodă primară. Această constatare a justificat trecerea către Deep Learning.

### 2.2 Arhitectura CNN Xception

Xception utilizează Depthwise Separable Convolutions pentru eficiență computațională superioară. Arhitectura pre-antrenată pe ImageNet (1.4M imagini) este adaptată prin Transfer Learning pe 100.000 imagini deepfake/reale.

## 3. Metodologie și arhitectură

Sistemul combină trei componente complementare:

1. **FFT (Analiză Spectrală):** Oferă explicabilitate vizuală
2. **Random Forest:** Baseline clasic (5 features extrase manual)
3. **CNN Xception:** Componentă decisivă (Transfer Learning)

### 3.1 Dataset și platformă de antrenare

**Date utilizate:**
- Random Forest: 2.041 imagini (training local)
- CNN: 100.000 imagini din "140k Real and Fake Faces" [4]
- Split: 80% training (80.000) / 20% validation (20.000)

| | |
|:---:|:---:|
| ![Imagine reală](/images/deepfake-detector/real/exemplu_real.jpg) | ![Imagine deepfake](/images/deepfake-detector/fake/exemplu_fake.jpg) |
| *Imagine reală — fotografie autentică* | *Imagine deepfake — generată cu AI* |

**Infrastructură:**
Datorită limitărilor hardware locale (placă video RTX 3050 4GB cu incompatibilități CUDA), antrenamentul s-a desfășurat în Google Colab pe accelerator Tesla T4 (16GB VRAM), reducând timpul de la 40+ ore (CPU local) la aproximativ 2 ore/epocă.

### 3.2 Strategie Transfer Learning

**Faza 1 (Base Frozen):**
- Learning Rate: 0.001
- Loss: Binary Crossentropy
- Optimizator: Adam
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Faza 2 (Fine-Tuning):** Unfreeze ultimele 20 straturi, LR=0.0001 (planificată, neexecutată din cauza expirării sesiunii Colab)

## 4. Implementare și interfață

### 4.1 Aplicația web Streamlit

Sistemul este disponibil public la:
**[https://deepfake-detector-tcsivtstcsitcsi.streamlit.app/](https://deepfake-detector-tcsivtstcsitcsi.streamlit.app/)**

**Deployment:** Streamlit Cloud cu Python 3.11, TensorFlow 2.15.1, model CNN (80MB) încărcat pe GitHub.

**Features Explainable AI:**
- Vizualizare spectru FFT 2D și profil radial
- Interpretare semantică prin OpenAI GPT-4o
- Predicție CNN cu breakdown probabilități
- Analiză comparativă între metode (FFT/RF/CNN)

### 4.2 Analize suplimentare

![Spectru 2D FFT](/images/deepfake-detector/fft_2d_spectrum.png)
*Figura 2: Spectru 2D FFT — vizualizare componente de frecvență*

| | |
|:---:|:---:|
| ![Color Histogram](/images/deepfake-detector/color_histogram.png) | ![Gradient Magnitude](/images/deepfake-detector/gradient_magnitude.png) |
| *Color Histogram — distribuție canale RGB* | *Gradient Magnitude — hartă de detalii* |

| | |
|:---:|:---:|
| ![Noise Pattern](/images/deepfake-detector/noise_pattern.png) | ![EXIF Metadata](/images/deepfake-detector/exif_metadata.png) |
| *Noise Pattern — analiză zgomot* | *EXIF Metadata — informații cameră* |

## 5. Rezultate experimentale

### 5.1 Comparație performanță

| Metodă | Acuratețe | AUC | Recall | Observații |
|--------|-----------|-----|--------|------------|
| Random Forest | 54.0% | — | 20% (Fake) | Bias sever spre Real |
| **CNN Ep. 6 (Best)** | **75.20%** | **0.8286** | **74.84%** | **Optim după LR decay** |
| CNN Epoch 3 | 74.67% | 0.8273 | 78.90% | Peak inițial |

### 5.2 Evoluția antrenamentului

| Epoca | Train Acc | Val Acc | Val AUC | Val Recall | Status |
|-------|-----------|---------|---------|------------|--------|
| 1 | 66.65% | 73.52% | 0.8150 | 78.04% | Salvat |
| 2 | 70.46% | 74.43% | 0.8231 | 79.46% | Salvat |
| 3 | 70.45% | 74.67% | 0.8273 | 78.90% | Peak inițial |
| 4 | 70.09% | 73.23% | 0.8248 | 83.37% | Val drop |
| 5 | 70.06% | 74.40% | 0.8279 | 80.73% | LR→0.0005 |
| **6** | **70.26%** | **75.20%** | **0.8286** | **74.84%** | **Best final** |

**Observații:**
- **Convergență rapidă:** 73.5% acuratețe în prima epocă
- **Efectul ReduceLROnPlateau:** LR scăzut la 0.0005 în Epoca 5 → îmbunătățire Val Accuracy la 75.2% (Epoca 6)
- **Timp/epocă:** ~33-37 minute pe Tesla T4 (2017-2212s)
- **Limitare infrastructură:** Sesiune Colab expirată în Epoca 9/10 (tier gratuit)

### 5.3 Analiză comparativă

- **Random Forest:** Generalizare slabă (54%), feature engineering manual insuficient
- **CNN Xception:** Generalizare puternică (75.2%), învățare automată de features, AUC ridicat (0.829)

## 6. Concluzii și direcții viitoare

### Realizări

1. **Sistem funcțional complet:** Deployment public pe Streamlit Cloud
2. **Performanță validată:** 75.2% accuracy, AUC 0.829 pe 20.000 imagini de validare
3. **Explicabilitate:** Integrare GPT-4o pentru interpretare grafice și valori
4. **Comparație metodologică:** Evidențierea limitărilor FFT și Random Forest vs. CNN

### Limitări

**Infrastructură:**
- Antrenament incomplet (6/20 epoci planificate) din cauza expirării sesiunii Colab gratuite
- Model final suboptimal — necesită Faza 2 (fine-tuning) pentru 88-92% accuracy

![Sesiune Colab expirată](/images/deepfake-detector/colab_epoch_limitation.png)
*Figura 3: Sesiune Google Colab expirată în timpul Epocii 9 — limitare infrastructură gratuită*

**Metodologice:**
- Detecție limitată la imagini statice (nu video)
- Performanță necunoscută pe deepfake-uri sofisticate (post-procesate profesional)
- Dataset cu imagini din 2020-2024 — imaginile AI generate cu software de ultima generație pot fi mai complicat de distins

### Direcții viitoare

- **Finalizare training:** Completare Faza 1 (epoci 7-10) + Faza 2 (fine-tuning)
- **Analiză video:** Extindere la detectare deepfake-uri video (frame-by-frame + temporal consistency)
- **Combinarea mai multor metode:** Combinare Xception + EfficientNet + Vision Transformer

---

## Bibliografie

1. Chollet, F., "Xception: Deep Learning with Depthwise Separable Convolutions", *CVPR*, 2017.
2. Durall, R., et al., "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions", *CVPR*, 2020.
3. Karras, T., et al., "Analyzing and Improving the Image Quality of StyleGAN", *CVPR*, 2020.
4. Kaggle, "140k Real and Fake Faces Dataset", 2023.
5. OpenAI, "GPT-4 Technical Report", 2024.
6. Streamlit Inc., "Streamlit Community Cloud Documentation", 2025.

{{< /lang >}}
