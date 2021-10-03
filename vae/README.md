# Compression with VAE using compressai

Source: https://github.com/InterDigitalInc/CompressAI

## Setup
Install with requirements.txt:
````bash
pip3 install -r requirements.txt
````
## Training
Use the training scripts in ``./trainings/``:
There are two scrips, one for image training, and one for point cloud training. Point Cloud training requires transformed pointclouds in ``.npz`` format.

Example command for training:
````bash
python3 trainings/train.py \
    -m bmshj2018-hyperprior \
    --train /disk/ml/datasets/KITTI/object/data/training/image_2 \
    --test /disk/ml/datasets/KITTI/object/data/testing/image_2 \
    --epochs 100 -lr 1e-4 \
    --batch-size 16 \
    --cuda \
    --save \
    --name hyperprior_5_01_1152-384_2021_06_28 5 \
    --quality 5 \
    --lambda 0.01 
````
See ``python3 trainings/train.py --h`` for help.

## Evaluation
Evaluation was done with several scripts in ``./notebooks/`` and ``./evaluations``. Note that code contains a lot of hardcoded folder and image paths!


