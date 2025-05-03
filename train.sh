#!/bin/bash

## Train from scratch
# python train.py --gpus 0,1,2,3 --config /zpool-00/data/jerryzhang/SW/FunCodec/egs/LibriTTS/text2speech_laura/conf/text2audio_codec_lm_nq2_uni_rel_pos.yaml


python train.py --gpus 0,1,2,3,4,5,6,7 --config config/conf.yaml

