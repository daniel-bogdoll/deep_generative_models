# Offline Pipeline Evaluation
**Please follow installation instructions in .``/vae/README.md`` first!**

This folder contains various scripts and notebooks for VAE, GAN and JPEG comparison.
This includes:
- Evaluation with KITTI dataset
- Object detection evaluation
- Point cloud KITTI evaluation

For each there are to files available: one Python script and one Jupyter notebook. The Python script can be executed and tests a preset dataset (e.g. KITTI) with given (trained) models. The results will then be saved in corresponding folders as .csv files. The .csv files can then be imported and its data evaluated using the respective Jupyter notebooks.

Apart from that, some experiments can be done using 
:
- ``CompressAI Experiments.ipynb``
- ``PointClouds Experiments.ipynb``