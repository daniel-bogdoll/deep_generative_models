# VAE Compression Research using CompressAI

Implementation of the Paper 
*Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston: "Variational image compression with a scale hyperprior"*.

The repository was customized to own needs.

## Training
For training images use the training script provided in ``./trainings/trainin.py``.
Example:
````bash
python3 trainings/train.py -m bmshj2018-hyperprior \
    --train /disk/ml/datasets/KITTI/object/data/training/image_2 \
    --test /disk/ml/datasets/KITTI/object/data/testing/image_2 \
    --epochs 100 \
    -lr 1e-4 \
    --batch-size 16 \
    --cuda \
    --save \
    --name hyperprior_5_0.01_big_2021_06_28  \
    --quality 5 \
    --lambda 0.01
````
Training chackpoints and results will be saved in ``training/NAME``. An additional ``.json`` file with all training parameters is saved, too.

For training with pointclouds use ``./rainings/train_pointclouds.py`` accordingly.

After training the models run:
````bash
python3 -m compressai.utils.update_model --architecture bmshj2018-hyperprior PATH_TO_CHECKPOINT
````
to make model inference ready.

# Evaluation
See ``./evaluations``

# References: 
- Paper: https://arxiv.org/abs/1802.01436
- CompressAI: https://github.com/InterDigitalInc/CompressAI
- Alternative Implementation of used Paper: https://github.com/liujiaheng/compression

