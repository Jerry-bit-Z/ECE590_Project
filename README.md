# 🎧 ECE590 Final Project: Lightweight Audio Transformer for Flexible Multimodal Speech Generation

Author: Miantong Zhang  
Date: April 2025  

---

## 🧠 Project Overview

This project aims to **reproduce and optimize the open-source LauraGPT model** for real-time audio-to-audio and multimodal generation tasks. The focus is on **lightweight model design**, **multiformat inference**, and **efficient deployment** under limited hardware resources.

Key objectives:
- Compress model size (from ~77MB to ~20MB)
- Support diverse generation modes:  
  - `Speech → Speech`  
  - `Speech + Text → Speech`  
  - `Speech → Speech + Text`  
- Train with `speech + text`; optionally infer with `text only`
- Enable deployment on Duke GPU server (8×2080Ti) and Kunshan A100 cluster

---

## ⚙️ Methodology

### 🧩 Stage 1: Reproduction & Optimization
- Reproduce [LauraGPT](https://github.com/Beilong-Tang/laura_gpt)
- Reduce model size for real-time inference
- Integrate discrete EnCodec tokens

### 🌐 Stage 2: Functional Extension
- Implement multimodal input fusion
- Enable multiple inference formats
- Evaluate text-modality enhancement

---

## 🧪 Datasets & Resources

- **Datasets**: VCTK, LibriSpeech, EnCodec-tokenized audio
- **Servers**:
  - Duke GPU Server (8×RTX 2080 Ti, 376GB RAM)
  - Kunshan Supercomputing Platform (A100/V100, via faculty)

---

## 🌟 Innovation Highlights

- Multiformat generation for hybrid agent design
- LLM-agent compatibility (text-only inference)
- Lightweight deployment for real-world settings

---

## ⏳ Timeline (Mar 31 – May 2)

| Week | Task |
|------|------|
| 1 | Reproduction & debugging |
| 2 | Lightweight adaptation |
| 3 | Multimodal integration |
| 4 | Training + small-batch eval |
| 5 | Final validation & demo |

---

## 🧪 (Optional) Speech Separation (Exploratory)

Exploring speaker-conditioned separation using short prompts (LibriMix / WSJ0-2mix). This direction aligns with the contextual capability of audio LLMs, and is treated as **auxiliary** and **low-priority**.

---

## 🔗 References

- [R1] InternLM Team. LauraGPT. GitHub, 2024.  
- [R2] Zeghidour et al. EnCodec: High Fidelity Neural Audio Compression, ICLR 2023  
- [R3] Vaswani et al. Attention is All You Need, NeurIPS 2017  

---

## 📁 Folder Structure

```bash
SW/
└── ECE590_Project/
    ├── data/            # Tokenized input datasets
    ├── notebooks/       # Training, inference, visualization
    ├── scripts/         # Model/repo setup & utility scripts
    ├── models/          # Checkpoints, architecture files
    ├── results/         # Generated samples, metrics, plots
    └── README.md
