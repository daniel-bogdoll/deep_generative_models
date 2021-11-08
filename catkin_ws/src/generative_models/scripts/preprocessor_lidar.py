#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch

from datetime import datetime

from lidar_compression.util.pointcloud_sort import (add_spherical_coordinates, 
                                                    remove_distant_and_close_points,
                                                    sort_points)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class LidarPreprocessor():
    def __init__(self):        
        self.pc_height = rospy.get_param("pc_height")
        self.pc_width = rospy.get_param("pc_width")
        self.out_fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        rospy.init_node(name="lidar_preprocessor_node", anonymous=False)
        self.subscriber = rospy.Subscriber(name="/carla/ego_vehicle/lidar",
                                            data_class=PointCloud2,
                                            callback=self.preprocessor_callback)
        self.publisher = rospy.Publisher(name="preprocessed_lidar/", data_class=PointCloud2, queue_size=10)

    def preprocessor_callback(self, in_cloud):
        global timing_df
        id = in_cloud.header.seq
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_preprocess = torch.cuda.Event(enable_timing=True)
        end_preprocess = torch.cuda.Event(enable_timing=True)
        
        start.record()
        gen_raw_points = pc2.read_points(in_cloud)   
        arr_raw_points = np.array(list(gen_raw_points))[:,:3]

        start_preprocess.record()
        sorted_cloud = sort_points(add_spherical_coordinates(remove_distant_and_close_points(arr_raw_points)),
                                                             num_cols=self.pc_width,
                                                             num_rows=self.pc_height)
        end_preprocess.record()
  

        out_header = Header(in_cloud.header.seq, rospy.Time.now(), in_cloud.header.frame_id)
        sorted_cloud = sorted_cloud.reshape((-1,3))
        out_pc = pc2.create_cloud(out_header, self.out_fields, sorted_cloud)
        self.publisher.publish(out_pc)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_preprocess = start_preprocess.elapsed_time(end_preprocess) # milliseconds
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_preprocess': elapsed_time_preprocess, 'timestamp': timestamp}, ignore_index=True)

def write_timing():
    global timing_df # maybe needed, try it out!    
    # timing_df.to_csv("SAVING_PATH")

if __name__ == "__main__":
    listener = LidarPreprocessor()    
    # rospy.on_shutdown(write_timing)
    rospy.spin()
