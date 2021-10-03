"""
usage: preprocess.py [-h] [--dir DIR] [--file FILE] [--dataset DATASET] [--write_dir WRITE_DIR] [--ratio_used RATIO_USED]

Utility for preprocessing lidar data.

optional arguments:
  -h, --help            show this help message and exit
  --dir DIR             Path to the folder containing the lidar file
  --file FILE           Name of the lidar file
  --dataset DATASET     Name of the dataset you want to use: {kitti, waymo}
  --write_dir WRITE_DIR
                        Indicating if you want to store the preprocessed point clouds as .npz files. Creates the path if it does not already exist.
  --ratio_used RATIO_USED
                        Only for Waymo. Ratio of scans that you want to convert, e.g., 1 means all scans and 0.25 means every fourth scan. Sampling is deterministic, which means the same scans
                        (and their names on disk) are used in subsequent runs if ratio_used does not change. Defaults to 1.
"""

import argparse
import numpy as np
import os
import tensorflow as tf
import traceback
from pathlib import Path
from tqdm import tqdm

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import lidar_compression.util.lidarutils as lidar
from lidar_compression.util.lidarexceptions import PointException, InvalidInputException, LineNumberException
from lidar_compression.util.pointcloud_sort import add_spherical_coordinates, remove_distant_and_close_points, sort_points

def _get_args():
    """Parses the CLI arguments.

    Returns
    -------
    Namespace
        Contains all arguments
    """
    parser = argparse.ArgumentParser(
        description = '''Utility for preprocessing lidar data.''',epilog="")    
    parser.add_argument('--dir', type=str, default=".", help='Path to the folder containing the lidar file')
    parser.add_argument('--file', type=str, default="", help='Name of the lidar file')
    parser.add_argument('--dataset', type=str, default="kitti", help='Name of the dataset you want to use: {kitti, waymo}')    
    parser.add_argument('--write_dir', type=str, default="", help='Indicating if you want to store the preprocessed point clouds as .npz files. Creates the path if it does not already exist.')
    parser.add_argument('--ratio_used', type=float, default=1, help='Only for Waymo. Ratio of scans that you want to convert, e.g., 1 means all scans and 0.25 means every fourth scan. Sampling is deterministic, which means the same scans (and their names on disk) are used in subsequent runs if ratio_used does not change. Defaults to 1.')

    return parser.parse_args()

def preprocess_write_kitti(in_dir:str, out_dir:str)->None:
    """Transforms the KITTI lidar scans into an ordered representation and writes it to disk as .npz files.
    It will not overwrite existing files in the out_dir by skipping them in order to save execution time.
    
    The output shape is cloud[rows][columns][channels], where rows correspond to elevation angle and columns to azimuth. This ordering ist necessary for doing convolutions on the point clouds.

    Parameters
    ----------
    in_dir:str
        Source of the KITTI binaries
    out_dir:str
        Target path for storing the converted npz files
    """
    file_count = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Created directory.")
    not_converted = Path(out_dir) / 'not_converted.txt'
    if not os.path.exists(not_converted):
        not_converted.touch()
    for filename in tqdm(os.listdir(in_dir), "Preprocessing and storing..."):
        out_filename = filename.split('.')[0] + '.npz'
        exists_npz = os.path.exists(Path(out_dir) / out_filename)
        if not exists_npz and filename.endswith(".bin"):
            cloud_3d = lidar.read_raw_velo(Path(in_dir) / filename)
            try:
                cloud_2d, _ = lidar.process_velo_kitti(cloud_3d, 512)            
            except PointException as e:                
                with open(Path(out_dir) / 'not_converted.txt', 'a') as not_converted:
                    content = e.args[0]
                    print("File not converted, because shape is wrong. Shape = " + content['shape'])
                    not_converted.write('{},{},PointException\n'.format(filename, content['points']))
                    continue
            except InvalidInputException as e:
                with open(Path(out_dir) / 'not_converted.txt', 'a') as not_converted:
                    content = e.args[0]
                    print("File not converted, because no condition in get_quadrant() fitted. Point shape = {}, point = {}".format(content['shape'], content['point']))
                    not_converted.write('{},{},InvalidInputException\n'.format(filename, content['point']))
                    continue
            except LineNumberException as e:
                with open(Path(out_dir) / 'not_converted.txt', 'a') as not_converted:
                    content = e.args[0]
                    print("File not converted: {} Number of lines = {}".format(content['message'], content['lines']))
                    not_converted.write('{},{},LineNumberException\n'.format(filename, content['lines']))
                    continue                  
            # print("\nShape: " + str(cloud_2d.shape))
            # print("\nOut file = " + out_filename)
            out_file = Path(out_dir) / out_filename            
            np.savez(out_file, kitti_cloud_2d=cloud_2d)
            file_count += 1    
        else:
            print("File already exists: " + filename.split('.')[0] + '.npz')
    print("{} KITTI-files written.".format(file_count))
    print()

def preprocess_write_waymo(in_dir:str, out_dir:str, ratio_used:float=1.0)->None:
    """Transforms the Waymo lidar scans into an ordered representation and writes them to disk as .npz files. Each point cloud gets a .npz file.
    It will not overwrite existing files in the out_dir by skipping them in order to save execution time.
    
    The output shape is cloud[rows][columns][channels], where rows correspond to elevation angle and columns to azimuth. This ordering ist necessary for doing convolutions on the point clouds.

    Parameters
    ----------
    in_dir : str
        Source of the KITTI binaries
    out_dir : str
        Target path for storing the converted npz files
    ratio_used : float
        Ratio of scans that you want to convert, e.g., 1 means all scans and 0.25 means every fourth scan. Sampling is deterministic, which means the same scans (and their names on disk) are used in subsequent runs if ratio_used does not change. Defaults to 1.
    """
    file_count = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Created directory: {out_dir}")
    not_converted = Path(out_dir) / 'not_converted.txt'
    if not os.path.exists(not_converted):
        not_converted.touch()
    for filename in tqdm(os.listdir(in_dir), "Preprocessing and storing..."):
        dataset = tf.data.TFRecordDataset(Path(in_dir) / filename)
        idx_in_dataset = -1
        try:
            for data in dataset:
                # Each data is a point cloud
                idx_in_dataset += 1
                if idx_in_dataset % (1 // ratio_used) != 0:
                    # Only use a fraction of the scans.
                    # But count index anyways, because then each data always has the same index,
                    # when running the script multiple times 
                    continue
                out_filename = filename.split('.')[0] + '-' + str(idx_in_dataset) + '.npz'
                exists_npz = os.path.exists(Path(out_dir) / out_filename)
                if not exists_npz and filename.endswith(".tfrecord"):            
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))                
                    (range_images, camera_projections,
                        range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
                    points, _ = frame_utils.convert_range_image_to_point_cloud(
                        frame,
                        range_images,
                        camera_projections,
                        range_image_top_pose)
                    # 3d points in vehicle frame. Not sorted.
                    cloud = np.concatenate(points, axis=0)
                    try:
                        sorted_cloud = sort_points(add_spherical_coordinates(remove_distant_and_close_points(cloud)),
                                                   num_cols=512,
                                                   num_rows=64)
                    except Exception:
                        with open(Path(out_dir) / 'not_converted.txt', 'a') as not_converted:                    
                            print("File not converted.")
                            not_converted.write('{}\n'.format(out_filename))
                            traceback.print_exc()
                            continue                
                    out_file = Path(out_dir) / out_filename
                    np.savez(out_file, sorted_waymo_cloud=sorted_cloud)
                    file_count += 1
                else:
                    print("File already exists: {}".format(out_filename))                
        except:
            print("Could not load tfrecord, skipping...")                
    print("{} Waymo-files written.".format(file_count))
    print()

if __name__ == '__main__':
    args = _get_args()    
    # Format: raw[0] = (x,y,z,reflectivity)
    if args.dataset == 'kitti':
        print("Using KITTI data")
        if args.write_dir != "":
            # transform and write to disk
            preprocess_write_kitti(args.dir, args.write_dir)
        else:
            # for testing purposes
            cloud_3d = lidar.read_raw_velo(Path(args.dir) / args.file)
            print("3D Cloud shape: " + str(cloud_3d.shape))
            cloud_2d, _ = lidar.process_velo_kitti(cloud_3d, 512)
            # Format: cloud[rows][columns][channels]
            # columns = points_per_layer        
            print("2D Cloud shape: " + str(cloud_2d.shape))
            print(cloud_2d[-1][-1])
            # Should remove reflectivity at some point
    elif args.dataset == 'waymo':
        print("Using Waymo data")
        if args.write_dir != "":
            # Transform and write to disk
            preprocess_write_waymo(args.dir, args.write_dir, args.ratio_used)            
        else:
            # For testing purposes
            file = args.dir + args.file
            dataset = tf.data.TFRecordDataset(file)
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                break
            (range_images, camera_projections,
                range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose)
            # 3d points in vehicle frame.
            cloud_3d = np.concatenate(points, axis=0)
            
            cartesian_spherical_cloud = add_spherical_coordinates(cloud_3d)
            sorted_cloud = sort_points(cartesian_spherical_cloud)
            print("Length of output: " + str(len(sorted_cloud)))            
            print("Type of output: " + str(type(sorted_cloud)))
            print("Shape of output: " + str(sorted_cloud.shape))
            # print("[0] of output: " + str(sorted_cloud[0]))
            print("Unique values: " + str(len(np.unique(sorted_cloud))))
            print("Number of non-zero-values: " + str(np.count_nonzero(sorted_cloud)))
            print("Number of zero-values: " + str(sorted_cloud.size - np.count_nonzero(sorted_cloud)))
            print("Ratio of zero-values w.r.t cloud size: "
                + str((sorted_cloud.size - np.count_nonzero(sorted_cloud)) / sorted_cloud.size))
            