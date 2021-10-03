import numpy as np
import unittest
import tensorflow as tf
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from lidar_compression.util.pointcloud_sort import add_spherical_coordinates, remove_distant_and_close_points, sort_points
from lidar_compression.util.pointcloud_sort_optimizations import sort_points_theta_phi_naive

class WaymoConversionTestCase(unittest.TestCase):

    def test_polar_transformation(self):
        # print("** Testing polar transformation")
        cartesian_points= np.array([[1,2,3], [-1,-2,-3]])
        cartesian_spherical_points = add_spherical_coordinates(cartesian_points)
        
        self.assertEqual(cartesian_points.shape[0], cartesian_spherical_points.shape[0])
        self.assertEqual(cartesian_spherical_points.shape[1], 6)
        self.assertAlmostEqual(cartesian_spherical_points[0][3], 3.7417, 4)
        self.assertAlmostEqual(cartesian_spherical_points[0][4], 0.6405, 4)
        self.assertAlmostEqual(cartesian_spherical_points[0][5], 1.1071, 4)
        self.assertAlmostEqual(cartesian_spherical_points[1][3], 3.7417, 4)
        self.assertAlmostEqual(cartesian_spherical_points[1][4], 2.5011, 4)
        self.assertAlmostEqual(cartesian_spherical_points[1][5], -2.0344 + 2*np.pi, 4)
    
    def test_sorting(self):
        # print("** Testing Waymo sorting")
        num_rows = 2
        num_cols = 2
        cartesian_points= np.array([[1,2,3], [-1,-2,-3], [5,6,3], [-5,3,10], [0,0,0], [10,10,10]], dtype=np.float)
        cartesian_spherical_points = add_spherical_coordinates(cartesian_points)
        sorted = sort_points(cartesian_spherical_points, num_cols=num_cols, num_rows=num_rows)
        sorted_naive = sort_points_theta_phi_naive(cartesian_spherical_points, num_cols=num_cols, num_rows=num_rows)
        
        r0_c0_values = np.array([cartesian_points[0][0:3], cartesian_points[2][0:3], cartesian_points[4][0:3],
                                                                                        cartesian_points[5][0:3]])
        r0_c1_values = np.array([cartesian_points[3]])
        r1_c0_values = np.zeros((1,3))
        r1_c1_values = np.array([cartesian_points[1]])
        
        r0_c0 = np.array([r0_c0_values[:,0].mean(), r0_c0_values[:,1].mean(), r0_c0_values[:,2].mean()])
        r0_c1 = np.array([r0_c1_values[:,0].mean(), r0_c1_values[:,1].mean(), r0_c1_values[:,2].mean()])
        r1_c0 = np.array([r1_c0_values[:,0].mean(), r1_c0_values[:,1].mean(), r1_c0_values[:,2].mean()])
        r1_c1 = np.array([r1_c1_values[:,0].mean(), r1_c1_values[:,1].mean(), r1_c1_values[:,2].mean()])
        res = np.concatenate([r0_c0, r0_c1, r1_c0, r1_c1]).reshape((2,2,3))
        
        self.assertEqual(sorted.shape, (num_rows,num_cols,3))
        self.assertTrue(isinstance(sorted, np.ndarray))
        self.assertEqual(sorted.dtype, "float64")
        self.assertTrue(np.array_equal(res, sorted))
        self.assertTrue(np.array_equal(res, sorted_naive))

    def test_optimization(self):
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
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        # 3d points in vehicle frame.
        cloud_3d = np.concatenate(points, axis=0)

        cartesian_spherical_cloud = add_spherical_coordinates(cloud_3d)
        sorted_cloud = sort_points(cartesian_spherical_cloud)
        sorted_cloud_naive = sort_points_theta_phi_naive(cartesian_spherical_cloud)

        self.assertTrue(np.array_equal(sorted_cloud, sorted_cloud_naive))
    
    def test_outlier_removal(self):
        points_xy = np.zeros((10,2))
        points_z = np.zeros((10,1))
        for i in range(1,11):
            points_xy[i-1] = np.array([i,i])
        for i in range(10,0,-1):
            points_z[10-i] = np.array([i])
        points = np.concatenate([points_xy, points_z], axis=1)
        # We calculated this by hand, it's correct.
        correct_result = points[2:8]
        # All points containing any value (i.e., x, y or z) greater than 8 should be thrown out
        # Code for trying out different percentiles (use python shell or notebook):
        #    result_xy = np.zeros((10,2))
        #    result_z = np.zeros((10,1))
        #    for i in range(1,11):
        #        result_xy[i-1] = np.array([i,i])
        #    for i in range(10,0,-1):
        #        result_z[10-i] = np.array([i])
        #    result = np.concatenate([result_xy, result_z], axis=1)
        #    result
        reduced_points = remove_distant_and_close_points(points, min_percentile=0, max_percentile=78)
        self.assertTrue(np.array_equal(reduced_points, correct_result))
        
if __name__ == "__main__":
    unittest.main()