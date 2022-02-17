# pct_code
Generate pseudo CT images form PET BP and PET NAC images based on GAN.

Currently the code is incomplete, and we will soon fix it.

## Requirement
pytorch=1.4


## Trainning 
```python
python train.py --dataroot /yaop/raofan/3d-pix2pix-CycleGAN-master/dataset/ --phase train --loadSize 256 --fineSize 256 --depthSize 8 --input_nc  2 --output_nc 1 --which_model_netD basic --which_model_netG unet_copy --dataset_mode  myaligned --model pix2pix3d --save_latest_freq 500 --batchSize 4 --lr 0.0001 --checkpoints_dir /yaop/raofan/3d-pix2pix-CycleGAN-master_v2/checkpoints_6 --pool_size 500 --gpu_ids 1 --lambda_A 100 --load_G_net empty --load_D_net empty
```

## Evaluation
```python
python test.py
```

## Citation 
Fan Rao, Zhuoxuan Wu, Lu Han, Bao Yang, Weidong Han, Wentao Zhu. Delayed PET Imaging Using Image Synthesis Network and Nonrigid Registration without Additional CT Scan. Medical Physics. 2022 (May not be available online now)
