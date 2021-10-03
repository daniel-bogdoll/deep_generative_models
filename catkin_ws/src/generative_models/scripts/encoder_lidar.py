#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import torch
import rospy
import sensor_msgs.point_cloud2 as pc2

from datetime import datetime

from compressai.zoo import bmshj2018_hyperprior
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from torchvision import transforms

from generative_models.msg import EncodedLidar, BytesList, IntList

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class Encoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")
        self.pc_height = rospy.get_param("pc_height")
        self.pc_width = rospy.get_param("pc_width")
        
        # ROS
        rospy.init_node(name="encoder_node", anonymous=False) 
        
        # TODO Remote addresse http
        self.subscriber = rospy.Subscriber(name="/preprocessed_lidar",
                                            data_class=PointCloud2,
                                            callback=self.encoder_callback)
        
        self.publisher = rospy.Publisher(name="encoded_lidar/", data_class=EncodedLidar, queue_size=10)

        # CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu' # To run on a single workstation together with Carla
        rospy.logfatal(self.device)

        # VAE model
        self.net = bmshj2018_hyperprior(quality=1, pretrained=False)
        self.net.load_state_dict(torch.load(model_weights_path))
        self.net.eval()
        self.net.to(self.device)
        rospy.logfatal('Parameters: {}'.format(sum(p.numel() for p in self.net.parameters())))

    def encoder_callback(self, in_cloud):
        global timing_df
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_encoder = torch.cuda.Event(enable_timing=True)
        end_encoder = torch.cuda.Event(enable_timing=True)

        start.record()
        
        id = in_cloud.header.seq        
        gen_cloud = pc2.read_points(in_cloud)        
        cloud = np.array(list(gen_cloud)).reshape((self.pc_height, self.pc_width, 3))        
        # Create tensor and add dimension
        tensor = transforms.ToTensor()(cloud).unsqueeze(0).to(self.device, dtype=torch.float)

        
        start_encoder.record()
        encoded_cloud = self.encode(tensor)
        end_encoder.record()
        
        
        strings_ros = []
        for b_list in encoded_cloud['strings']:
            bytes_list_ros = []
            for byte in b_list:
                ints = []
                for i in byte:
                    ints.append(i)
                bytes_list_ros.append(IntList(ints))                
            strings_ros.append(BytesList(bytes_list_ros))
        
        out_encoded_cloud = EncodedLidar() 
        out_encoded_cloud.header = Header(in_cloud.header.seq, rospy.Time.now(), in_cloud.header.frame_id)
        out_encoded_cloud.strings = strings_ros
        out_encoded_cloud.strings_shape = list(encoded_cloud['shape'])
        out_encoded_cloud.original_size = [(in_cloud.height), (in_cloud.width)]

        self.publisher.publish(out_encoded_cloud)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_compression = start_encoder.elapsed_time(end_encoder) # milliseconds
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_compression': elapsed_time_compression, 'timestamp': timestamp}, ignore_index=True)
        
        
    def encode(self, img_tensor):
        with torch.no_grad():
            # [JJ] type(output_encoder) = dict
            output_encoder = self.net.compress(img_tensor)
        return output_encoder

def write_timing():
    # global timing_df # maybe needed, try it out!    
    timing_df.to_csv("/fzi/ids/fa751/lidar/encoder.csv")

if __name__ == "__main__":
    listener = Encoder()    
    rospy.on_shutdown(write_timing)
    rospy.spin()
