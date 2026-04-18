# Multimodal Fake News Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
This repository contains the official implementation of the **Multimodal Fake News Detection** project. Moving beyond traditional text-only classifiers, this pipeline integrates **Text, Image, Audio, and Video** modalities to detect inconsistencies and fabricated content across diverse media formats. 

The architecture abandons basic concatenation methods in favor of advanced **Cross-Attention** and **Gated Fusion** mechanisms. By leveraging state-of-the-art transformer models for feature extraction, the system is highly resilient to noisy or missing modalities, establishing a robust standard for modern misinformation detection.

---

## Key Features

* **Comprehensive Multimodality:** Processes articles, images, audio clips, and deepfake videos.
* **Advanced Embedding Extraction:** Utilizes SBERT, CLIP, TIMM, Wav2Vec 2.0, and VideoMAE.
* **Dynamic Fusion:** * *Cross-Attention:* Maps contextual relationships across different media types.
    * *Gated Fusion:* Automatically weights modality importance, allowing the model to adapt when inputs (like video or audio) are missing.
* **Optimized Training:** Embeddings are pre-extracted and stored as `.npy` files to eliminate computational bottlenecks during training.

---

## System Architecture

### 1. Baseline Models
* **Text-Only Baseline:** A BiLSTM processing SBERT embeddings to capture sequential dependencies.
* **Bimodal Baseline:** A lightweight Text + Image model utilizing Gated Fusion.

### 2. Proposed Final Model
The core architecture processes four distinct data streams:
1.  **Text:** Sentence-BERT (`all-MiniLM-L6-v2`) -> 384-dim semantic vectors.
2.  **Image:** CLIP (Contrastive Language-Image Pretraining) & TIMM (EfficientNet/ViT).
3.  **Audio:** Wav2Vec 2.0 -> raw waveform and pitch representations.
4.  **Video:** VideoMAE -> spatial-temporal feature extraction.

These embeddings are integrated using Cross-Attention, followed by Gated Fusion:
`Fused = gate * modality_1 + (1 - gate) * modality_2`

The fused representation is passed through a fully connected transformer-head classifier to output the veracity label (`Real` or `Fake`).

---

## Datasets

The pipeline is trained and evaluated on a balanced, unified dataset comprising 43,131 samples from the following sources:
* **WELFake:** Text-based real vs. fake news.
* **FakeNewsNet:** Text + Image multimodal data.
* **FakeAVCeleb:** Audio + Video deepfake datasets.

*Note: The datasets are unified into a single tabular format and downsampled to balance the `text_only`, `text_image`, and `audio_video` categories.*

---
