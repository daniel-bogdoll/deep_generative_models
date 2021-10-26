# Compressing Sensor Data for Remote Assistance of Autonomous Vehicles using Deep Generative Models
## Code and supplementary materials

Repository of the paper **Compressing Sensor Data for Remote Assistance of Autonomous Vehicles using Deep Generative Models** at [ML4AD@NeurIPS 2021](https://ml4ad.github.io/).

## Online Pipeline

The left side of the video shows the ground truth video data from CARLA. On the right you see the VAE based reconstruction.

https://user-images.githubusercontent.com/19552411/135649380-de1865eb-b1d5-4852-ad9f-807104dca5a9.mp4

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