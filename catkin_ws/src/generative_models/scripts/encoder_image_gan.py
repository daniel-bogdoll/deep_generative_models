#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
from numpy.testing._private.utils import assert_array_equal
import pandas as pd
import torch
import rospy

from datetime import datetime

from sensor_msgs.msg import Image as ImageMsg
from torchvision import transforms
from PIL import Image

from generative_models.msg import EncodedImageGAN
from gan_compression import compress, load_model

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class Encoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")        
        # ROS
        rospy.init_node(name="encoder_node", anonymous=False)
        self.subscriber = rospy.Subscriber(name="/carla/ego_vehicle/rgb_front/image",
                                            data_class=ImageMsg,
                                            callback=self.encoder_callback)
        # GAN
        self.publisher = rospy.Publisher(name="GAN_encoded_imgs/", data_class=EncodedImageGAN, queue_size=10)
        # CUDA 
        # self.device = 'cpu' # To run on a single workstation together with Carla
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # GAN model
        self.gan = load_model(model_weights_path)

    def encoder_callback(self, in_img):
        global timing_df
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_compress = torch.cuda.Event(enable_timing=True)
        end_compress = torch.cuda.Event(enable_timing=True)
        
        timestamp_start = datetime.now().strftime("%H:%M:%S.%f")
        start.record()        
        id = in_img.header.seq
        # type(ImageMsg.data)=uint8 in ROS
        # Encoding from carla = bgra -> delete 'a' channel and invert array -> result is 'rgb'-ordering
        img_data = np.frombuffer(in_img.data, dtype=np.uint8).reshape((in_img.height, in_img.width, 4))[:,:,:-1][:,:,::-1]
        img = Image.fromarray(img_data, 'RGB')
        
        # Create tensor, add dimension, resize        
        # GAN
        tensor = transforms.ToTensor()(img.resize((256,256))).unsqueeze(0).to(self.device)

        start_compress.record()       
        encoded_img = self.encode(tensor)        
        end_compress.record()    
        
        out_encoded_img = EncodedImageGAN() 
        out_encoded_img.id = id
        out_encoded_img.original_size = [(in_img.height), (in_img.width)]
        out_encoded_img.hyperlatents_encoded = encoded_img.hyperlatents_encoded.tolist()
        out_encoded_img.latents_encoded = encoded_img.latents_encoded
        out_encoded_img.hyperlatent_spatial_shape = encoded_img.hyperlatent_spatial_shape
        out_encoded_img.batch_shape = encoded_img.batch_shape
        out_encoded_img.spatial_shape = encoded_img.spatial_shape
        out_encoded_img.hyper_coding_shape = encoded_img.hyper_coding_shape
        out_encoded_img.latent_coding_shape = encoded_img.latent_coding_shape
    
        self.publisher.publish(out_encoded_img)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_compression = start_compress.elapsed_time(end_compress) # milliseconds
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_compression': elapsed_time_compression, 'timestamp': timestamp, 'timestamp_start': timestamp_start}, ignore_index=True)

    def encode(self, img_tensor):
        with torch.no_grad():
            # [JJ] type(output_encoder) = dict
            # output_encoder = self.net.compress(img_tensor)
            output_encoder = compress(self.gan, img_tensor)
        return output_encoder

def write_timing():
    # global timing_df # maybe needed, try it out!    
    timing_df.to_csv("/fzi/ids/fa751/image_gan/encoder.csv")

if __name__ == "__main__":
    listener = Encoder()    
    rospy.on_shutdown(write_timing)
    rospy.spin()
