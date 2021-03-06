# Compression with VAE using compressai

Source: https://github.com/InterDigitalInc/CompressAI

## Setup
First, get the submodule:
````
git submodule update --init --recursive
````
Install all necessary libraries with requirements.txt:
```bash
pip3 install -r requirements.txt
```
For installing ``compressai`` you may follow instructions in ``./third_party/compressai`` folder.

## Training
Use the training scripts in `./trainings/`. There are two scrips, one for image training, and one for point cloud training. Point Cloud training requires transformed pointclouds in `.npz` format.

Example command for training with images:
```bash
python3 trainings/train.py \
    -m bmshj2018-hyperprior \
    --train PATH_FOR_TRAIN_IMAGES \
    --test PATH_FOR_TEST_IMAGES \
    --epochs 100 -lr 1e-4 \
    --batch-size 16 \
    --cuda \
    --save \
    --name DEFINE_A_NAME \
    --quality 5 \
    --lambda 0.01 
```
See `python3 trainings/train.py --h` for help.

Example command for training with point clouds:
```bash
python3 trainings/train_pointclouds.py \
    -m bmshj2018-hyperprior \
    --train PATH_FOR_TRAIN_POINT_CLOUDS \
    --test PATH_FOR_TEST_POINT_CLOUDS  \
    --epochs 100 \
    -lr 1e-4 \
    --batch-size 16 \
    --savefreq 20 \
    --cuda \
    --save \
    --name DEFINE_A_NAME \
    --quality 5 \
    --lambda 0.01 \
    --cloud_arg kitti_cloud_2d
```
See `python3 trainings/train_pointclouds.py --h` for help.

After training, models must be be made inference ready by:
````bash
python3 -m compressai.utils.update_model --architecture bmshj2018-hyperprior PATH_TO_TRAINED_MODEL_TAR
```` 

## Evaluation
Evaluation was done with several scripts in `evaluation/offline-pipeline` folder. Note that code contains some hardcoded folder and image paths!
