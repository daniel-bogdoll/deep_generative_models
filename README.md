# Compressing Sensor Data for Remote Assistance of Autonomous Vehicles using Deep Generative Models

Repository of the paper **[Compressing Sensor Data for Remote Assistance of Autonomous Vehicles using Deep Generative Models](https://arxiv.org/abs/2111.03201)** at [ML4AD@NeurIPS 2021](https://ml4ad.github.io/).

## Online Pipeline

The left side of the videos shows the ground truth data from CARLA. On the right you see the VAE based reconstructions. Videos are accelerated. For figure 6 in the paper the VAE model was trained with preprocessed lidar scans with a shape of 512x64 (same as for the images). This included some minor padding. After the submission we trained the VAE model with preprocessed lidar scans with a shape of 128x64 instead, which led to a much improved quality of the reconstructed pointclouds as you can see in the video.

https://user-images.githubusercontent.com/19552411/135649380-de1865eb-b1d5-4852-ad9f-807104dca5a9.mp4

https://user-images.githubusercontent.com/19552411/140888871-8678ced9-50f2-4da6-80ed-942d382bb239.mp4

## Repository Structure

See the specific folders for additional information.

```bash
.
├── catkin_ws       # ROS workspace for running the online pipeline
├── evaluation      # Evaluation results
├── gan             # The GAN we use
├── lidar           # Contains the lidar preprocessing package and supplementary code
├── paper-graphics  # Code that generates some of our graphics
└── vae             # The VAE we use
```

## Citation
If you find this code useful for your research, please cite our paper:
```
@article{Bogdoll_Compressing_2021_NeurIPS,
    author    = {Bogdoll, Daniel and Jestram, Johannes and Rauch, Jonas and Scheib, Christin and Wittig, Moritz and Z\"{o}llner, J. Marius},
    title     = {{Compressing Sensor Data for Remote Assistance of Autonomous Vehicles using Deep Generative Models}},
    journal   = {NeurIPS Conference on Neural Information Processing Systems Workshop on Machine Learning for Autonomous Driving (ML4AD)},
    year      = {2021},
}
```

## Acknowledgment
This work results partly from the KIGLIS project supported by the German Federal Ministry of
Education and Research (BMBF), grant number 16KIS1231.
