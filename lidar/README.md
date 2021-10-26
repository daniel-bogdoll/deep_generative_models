# Lidar Processing
`lidar_compression` contains the functions that perform the preprocessing of the lidar data. The basic approach is taken from Lucas Caccia et al.'s Paper [Deep Generative Modeling of LiDAR Data](https://arxiv.org/abs/1812.01180) on deep generative lidar compression. We reimplemented the approach in order to make it applicable to multiple datasets. Currently everything runs on CPU only.

## Install
In order to use this package, install the requirements and then the package itself.
1. `pip install -r requirements.txt`
1. `pip install .`

## Preprocses Single Point Clouds for Testing Purposes
- KITTI: run `python preprocess.py --dir "/disk/ml/datasets/KITTI/object/data/testing/velodyne" --file "000000.bin" --dataset "kitti"`
- Waymo: run `python preprocess.py --dir "/disk/vanishing_data/fa401/waymo/" --file "segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord" --dataset "waymo"`

## Preprocess and Write to .npz
- KITTI: `python preprocess.py --dir "/disk/ml/datasets/KITTI/object/data/testing/velodyne" --dataset "kitti" --write_dir "path/to/mlp_kitti_clouds"`
- Waymo: `python preprocess.py --dir "/disk/ml/datasets/waymo/perception/domain_adaptation/training" --dataset "waymo" --write_dir "/disk/vanishing_data/fa401/waymo-transformed/training" --ratio_used=0.05`

## Hint: Load npz
1. `file = open("003231.npz", 'rb')`
2. `data = np.load(file)`
3. KITTI: `cloud = data['kitti_cloud_2d']`, Waymo: `cloud = data['sorted_waymo_cloud']`
4. `print(cloud.shape)`
