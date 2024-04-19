CUDA_VISIBLE_DEVICES=0 python demo.py  --config config/vox-256.yaml \
    --driving_video video/input.mp4 --source_image video/src.png \
    --checkpoint 00000099-checkpoint.pth.tar --relative --adapt_scale --kp_num 15 \
    --generator Unet_Generator_keypoint_aware \
    --result_video video/output.mp4 --mbunit ExpendMemoryUnit --memsize 1 