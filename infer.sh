#!/bin/bash

## Infer from using conf_right ckpt
## TODO: change the setting in default config
python -u inference/infer.py --config_file config/conf.yaml \
  --model_file ckpt/conf/epoch49.pth \
  --output_dir output_one \
  --default_config inference/infer_one.yaml\
  --raw_inputs "I love Duke University. ECE590 is my favourite class. It's about large language models based on Transformer architecture.This project explores a novel text-to-audio generation system by integrating the FunCodec neural codec framework with a GPT-based language model.Using the LibriTTS speech dataset, we develop a lightweight text-to-speech (TTS) pipeline that converts text into discrete audio codec tokens and then into waveform audio.We implement both single-GPU training/inference and multi-GPU distributed (DDP) training to achieve high-quality speech synthesis. The proposed method is evaluated in terms of training/validation loss, token alignment accuracy, and subjective audio quality."\
  --tokenize_to_phone

## Infer one speech
# python -u inference/infer.py --config_file config/conf_right.yaml \
#  --model_file ckpt/conf_right/best.pth\
#  --output_dir /public/home/qinxy/bltang/laura_gpt/output_one\
#  --default_config inference/infer_one.yaml\
#  --raw_inputs "I love Duke Kunshan University."\
#  --tokenize_to_phone 
