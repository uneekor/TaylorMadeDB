```shell
python download.py \
--input_path ./tmDB.csv \
--output_path ./output
```

```shell
python crop.py \
--input_path ./TMMade.youtube/Jlp8G9paliw.mp4 \
--crop_ratio 0.62 \
--crop_direction left \
--output_path ./TMMade.stereo
```

```shell
python crop.py \
--input_path ./TMMade.youtube/Jlp8G9paliw.mp4 \
--crop_ratio 0.62 \
--output_path ./TMMade.stereo
```


```shell
python extract_rtmpose_aihub_v3.py \
--in_dir ./TMMade.stereo \
--det_config /home/jglee/projects/mmyolo/train_aihub/m2/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_4gpu.py \
--det_checkpoint /home/jglee/projects/mmyolo/train_aihub/m2/epoch_175.pth \
--pose_config /home/jglee/projects/mmpose/rtmpose_aihub/m4/rtmpose-m_8xb256-420e_aihub-256x192_42epoch.py \
--pose_checkpoint /home/jglee/projects/mmpose/rtmpose_aihub/m4/epoch_42.pth 
```
