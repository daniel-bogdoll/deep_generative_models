# GAN Compression

Implementation of the Paper [High-Fidelity Generative Image Compression](https://arxiv.org/abs/2006.09965)* by Fabian Mentzer, George Toderici, Michael Tschannen, and Eirikur Agustsson.

The repository was adapted to our own needs from the [original publication](https://hific.github.io/).

## Training
Instructions:
1. If needed, setup new dataset in `./src/helpers/datasets.py` as new BaseDataset and add it to DATASETS_DICT (existing Datasets = ["openimages", "cityscapes",  "jetimages", "kitti", "waymo"]). Adjust the paths for KITTI and Waymo images.
2. Check in default_config.py file for general settings and declare in args which Dataset should be selected. Additionally correct path to dataset must be set in DatasetPaths. 
3. Train initial autoencoding model
    - Example:`python train.py --model_type compression --regime high --gpu 0 --n_steps 1e6`
4. Train using full generator-discriminator loss with a warmstart on the previous Autoencoder
    - Example: `python train.py --model_type compression_gan --regime low --gpu 0 --n_steps 1e6 --warmstart --warmstart_ckpt ./experiments/kitti_compression_2021_06_02_15_57/checkpoints/kitti_compression_2021_06_02_15_57_epoch5_idx4680_2021_06_02_16\:37.pt`

Remarks:
- Training chackpoints will be saved in ``experiments/dataset_compression_gan_timestamp/checkpoints/`` (e.g. ``./experiments/kitti_compression_gan_2021_06_03_10_30/checkpoints/``) 
- While training there will be information about the training logged in ``experiments/dataset_compression_gan_timestamp/tensorboard/`` which can be checked in tensorboard - example:
```tensorboard --logdir ./experiments/waymo_comession_2021_07_12_19_59/tensorboard/ --port 2401```

## Inference
- Compression: `-i` specifies the folder where the images that should be compressed are. A Folder with reconstructions will be automatically created.
- Example: `python compress.py -i ./data/originals/ -ckpt ./experiments/kitti_compression_gan_2021_06_03_10_30/checkpoints/kitti_compression_gan_2021_06_03_10_30_epoch7_idx7484_2021_06_03_11:27.pt --save`



## References 
- Paper: https://arxiv.org/abs/2006.09965
- HIFIC: https://github.com/Justin-Tan/high-fidelity-generative-compression
- Original implementation: https://hific.github.io/
