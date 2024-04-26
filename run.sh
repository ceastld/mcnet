#!/bin/bash

# 定义变量
name="zcl"
mkdir -p video/mcnet/zcl1

CUDA_VISIBLE_DEVICES=5 python demo.py  --config config/vox-256.yaml \
    --driving_video video/gyd/frontal1.mp4 \
    --source_image video/ids/zcl/zcl_ffhq.jpg \
    --result_video video/mcnet/zcl1/output.mp4 \
    --checkpoint 00000099-checkpoint.pth.tar --relative --adapt_scale --kp_num 15 \
    --generator Unet_Generator_keypoint_aware --mbunit ExpendMemoryUnit --memsize 1
    