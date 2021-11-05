#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch

from datetime import datetime

from compressai.zoo import bmshj2018_hyperprior
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from generative_models.msg import EncodedLidar

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class Decoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")
        self.pc_height = rospy.get_param("pc_height")
        self.pc_width = rospy.get_param("pc_width")
        self.out_fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        # ROS
        rospy.init_node(name="decoder_node", anonymous=False)        
        self.subscriber = rospy.Subscriber(name="encoded_lidar/",
                                            data_class=EncodedLidar,
                                            callback=self.decoder_callback)
        self.publisher = rospy.Publisher(name="decoded_lidar/", data_class=PointCloud2, queue_size=10)

        # CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        # VAE model
        self.net = bmshj2018_hyperprior(quality=1, pretrained=False)
        self.net.load_state_dict(torch.load(model_weights_path))
        self.net.eval()
        self.net.to(self.device)
        self.net.update()
        
    def decoder_callback(self, in_encoded_cloud):
        global timing_df
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)  
        start_decompress = torch.cuda.Event(enable_timing=True)
        end_decompress = torch.cuda.Event(enable_timing=True)  
        
        start.record()
        
        id = in_encoded_cloud.header.seq     
        deserialized_latent_space = [[bytes(intList.intlist) for intList in string.byteslist] for string in in_encoded_cloud.strings]
      
        
        start_decompress.record()
        decoded_cloud = self.decode(deserialized_latent_space, in_encoded_cloud.strings_shape)
        end_decompress.record()

        
        out_header = Header(in_encoded_cloud.header.seq, rospy.Time.now(), in_encoded_cloud.header.frame_id)
        out_pc = pc2.create_cloud(out_header, self.out_fields, decoded_cloud.reshape((-1,3)))
        self.publisher.publish(out_pc)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_decompress = start_decompress.elapsed_time(end_decompress) # milliseconds
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_decompress': elapsed_time_decompress, 'timestamp': timestamp}, ignore_index=True)

    def decode(self, strings, strings_shape):
        with torch.no_grad():
            output_decoder = self.net.decompress(strings, strings_shape)        
        output_tensor = np.asarray(output_decoder['x_hat'].squeeze().cpu())
        # Gotta swap axes
        swapped = np.swapaxes(np.swapaxes(output_tensor, 0, 2), 0,1)        
        return swapped

def write_timing():
    # global timing_df # maybe needed, try it out!    
    # timing_df.to_csv("SAVING_PATH")

if __name__ == "__main__":
    listener = Decoder()    
    # rospy.on_shutdown(write_timing)
    rospy.spin()
