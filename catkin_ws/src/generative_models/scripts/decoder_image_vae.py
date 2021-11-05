#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import rospy
import std_msgs.msg
import torch

from datetime import datetime

from compressai.zoo import bmshj2018_hyperprior
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from torchvision import transforms

from generative_models.msg import EncodedImageVAE

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class Decoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")        
        # ROS
        rospy.init_node(name="decoder_node", anonymous=False)
        self.subscriber = rospy.Subscriber(name="encoded_imgs/",
                                            data_class=EncodedImageVAE,
                                            callback=self.decoder_callback)        
        self.publisher = rospy.Publisher(name="decoded_imgs/", data_class=ImageMsg, queue_size=10)
        # CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        # VAE model
        self.net = bmshj2018_hyperprior(quality=1, pretrained=False)
        self.net.load_state_dict(torch.load(model_weights_path))
        self.net.eval()
        self.net.to(self.device)

    def decoder_callback(self, in_encoded_img):
        global timing_df

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)        
        start_decompress = torch.cuda.Event(enable_timing=True)
        end_decompress = torch.cuda.Event(enable_timing=True)    
        
        start.record()
        timestamp_start = datetime.now().strftime("%H:%M:%S.%f")
        
        id = in_encoded_img.header.seq      
        deserialized_latent_space = [[bytes(intList.intlist) for intList in string.byteslist] for string in in_encoded_img.strings]
        
        start_decompress.record()
        decoded_img = self.decode(deserialized_latent_space, in_encoded_img.strings_shape)
        end_decompress.record()

        
        reconstructed_img = transforms.ToPILImage()(decoded_img.squeeze().cpu())
        out_img_original_size = reconstructed_img.resize(in_encoded_img.original_size)

        out_message = ImageMsg()
        h = Header(in_encoded_img.header.seq, rospy.Time.now(), in_encoded_img.header.frame_id)        
        out_message.header = h        
        out_message.height = out_img_original_size.height
        out_message.width = out_img_original_size.width        
        out_message.encoding = "rgb8" # [JJ] TODO verify
        out_message.is_bigendian = False
        out_message.step = 3 * out_img_original_size.width
        out_message.data = np.array(out_img_original_size).tobytes()        
        self.publisher.publish(out_message)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_decompress = start_decompress.elapsed_time(end_decompress) # milliseconds

        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_decompress': elapsed_time_decompress, 'timestamp': timestamp, 'timestamp_start': timestamp_start}, ignore_index=True)
        

    def decode(self, strings, strings_shape):
        with torch.no_grad():
            output_decoder = self.net.decompress(strings, strings_shape)        
        output_decoder['x_hat'].clamp_(0, 1)
        output_tensor = output_decoder['x_hat']
        return output_tensor

def write_timing():
    # global timing_df # maybe needed, try it out!    
    # timing_df.to_csv("SAVING_PATH")

if __name__ == "__main__":
    listener = Decoder()    
    # rospy.on_shutdown(write_timing)
    rospy.spin()
