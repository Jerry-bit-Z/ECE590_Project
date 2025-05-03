# FunCodec + LauraGPT TTS

**Author:** Miantong Zhang  
**Date:** May 2025

## Overview
Lightweight end-to-end text-to-audio system combining FunCodec neural codec with a GPT-style Transformer (12 layers, 8 heads, hidden 512; ~66 M params). Trained on LibriTTS using single-GPU (RTX 2080 Ti) and 8-GPU DDP setups. Achieves real-time (≈0.78×) inference and dev loss 7.04 / dev acc 0.141.

## Features
- **Codec-driven TTS:** phonemes → codec tokens → waveform  
- **Training:** 10 epochs single-GPU (5 h) / 50 epochs DDP (19.1 h)  
- **Inference:** Greedy decoding, ~3.9 s per 5 s utterance  

## Repository
SW/laura_gpt/ ├── config/ # conf.yaml, conf_right.yaml ├── inference/ # infer scripts & configs ├── log/ # training logs & plotting ├── output_one/ # example WAVs ├── train.py # trainer entrypoint ├── train.sh # launcher (single-GPU & DDP) └── utils.py # helpers

## Pre-trained Weights & Audio Samples

The pre-trained model weights and example WAV files are available on Google Drive:  
[Download weights & samples](https://drive.google.com/drive/folders/11Xx_OO-Z9miWQaGoYRuozsjg5_APVsDw?usp=drive_link)

## Quick Start
```bash
git clone https://github.com/Jerry-bit-Z/ECE590_Project.git
cd ECE590_Project/SW/laura_gpt
bash train.sh --config config/conf.yaml --gpus 0
bash inference/infer.sh inference/infer.yaml inputs.txt

