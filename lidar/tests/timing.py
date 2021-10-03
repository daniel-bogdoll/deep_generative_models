import numpy as np
import tensorflow as tf
import timeit
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from lidar_compression.util.pointcloud_sort import add_spherical_coordinates, sort_points
from lidar_compression.util.pointcloud_sort_optimizations import sort_points_theta_phi_optimized_intermediate, sort_points_theta_phi_naive, sort_points_theta_phi_optimized_numpy_only

if __name__ == "__main__":
    dir = "/disk/vanishing_data/fa401/waymo/"
    filename = "segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
    file = dir + filename
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

    number_spherical = 1000
    number_sort = 20
    print("Time for running add_spherical_coordinates() {} times: {}[s]".format(str(number_spherical),
                str(timeit.timeit(lambda: add_spherical_coordinates(cloud_3d), number=number_spherical))))
    print("Time for running sort_points() {} times: {}[s]".format(str(number_sort),
                 str(timeit.timeit(lambda: sort_points(cartesian_spherical_cloud), number=number_sort))))
    print("Time for running sort_points_theta_phi_naive() {} times: {}[s]".format(str(number_sort),
        str(timeit.timeit(lambda: sort_points_theta_phi_naive(cartesian_spherical_cloud),number=number_sort))))
    print("Time for running sort_points_theta_phi_optimized_intermediate() {} times: {}[s]".format(str(number_sort),
                    str(timeit.timeit(lambda: sort_points_theta_phi_optimized_intermediate(cartesian_spherical_cloud),
                                                                                                number=number_sort))))
    print("Time for running sort_points_theta_phi_optimized_numpy_only() {} times: {}[s]".format(str(number_sort),
                    str(timeit.timeit(lambda: sort_points_theta_phi_optimized_numpy_only(cartesian_spherical_cloud),
                                                                                                number=number_sort))))
