#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import rospy
import std_msgs.msg
import torch

from datetime import datetime

from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from torchvision import transforms

from generative_models.msg import EncodedImageGAN
from gan_compression import load_model, decompress

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])


class Decoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")
        # ROS
        rospy.init_node(name="decoder_node", anonymous=False)
        # GAN
        self.subscriber = rospy.Subscriber(name="GAN_encoded_imgs/",
                                            data_class=EncodedImageGAN,
                                            callback=self.decoder_callback)        
        self.publisher = rospy.Publisher(name="GAN_decoded_imgs/", data_class=ImageMsg, queue_size=20)
        
        # CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu' # if GPU has not enough memory

        # GAN model
        rospy.logfatal("GAN LOADING!")
        self.gan = load_model(model_weights_path)
        rospy.logfatal("GAN LOADED!")

    def decoder_callback(self, in_encoded_img):
        global timing_df
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        end_decompress = torch.cuda.Event(enable_timing=True)
        
        start.record()   
        decoded_img = self.decode(in_encoded_img)
        end_decompress.record()   


        reconstructed_img = transforms.ToPILImage()(decoded_img.squeeze().cpu())
        out_img_original_size = reconstructed_img.resize(in_encoded_img.original_size)

        out_message = ImageMsg()
        out_message.header = Header(in_encoded_img.id, rospy.Time.now(), "carla")
        out_message.height = out_img_original_size.height
        out_message.width = out_img_original_size.width
        out_message.encoding = "rgb8" 
        out_message.is_bigendian = False
        out_message.step = 3 * out_img_original_size.width
        out_message.data = np.array(out_img_original_size).tobytes()
            
        self.publisher.publish(out_message)

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_decompression = start.elapsed_time(end_decompress) # milliseconds
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_decompress': elapsed_time_decompression, 'timestamp': timestamp}, ignore_index=True)
        
    def decode(self, in_encoded_img):

        in_encoded_img.hyperlatents_encoded = np.asarray(list(in_encoded_img.hyperlatents_encoded)).astype(np.uint32)
        in_encoded_img.latents_encoded = np.asarray(list(in_encoded_img.latents_encoded)).astype(np.uint32)
        in_encoded_img.hyperlatent_spatial_shape = torch.empty(in_encoded_img.hyperlatent_spatial_shape[0], in_encoded_img.hyperlatent_spatial_shape[1]).size()

        with torch.no_grad():
            output_decoder = decompress(self.gan, in_encoded_img)      

        return output_decoder

def write_timing():
    # global timing_df # maybe needed, try it out!    
    # timing_df.to_csv("SAVING_PATH")

if __name__ == "__main__":
    listener = Decoder()    
    # rospy.on_shutdown(write_timing)
    rospy.spin()
