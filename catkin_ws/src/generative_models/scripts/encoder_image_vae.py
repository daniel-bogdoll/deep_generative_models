#!/usr/bin/env python3

#   This node transforms the velodyne pointclouds from their own frame to base_link frame
#   It republishes the transformed PCs to rosbag2kitti/<sensorname>

import numpy as np
import pandas as pd
import torch
import rospy

from compressai.zoo import bmshj2018_hyperprior
from datetime import datetime
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from torchvision import transforms
from PIL import Image

from generative_models.msg import EncodedImageVAE, BytesList, IntList

timing_df = pd.DataFrame(columns=['frameID', 'inference_time', 'timestamp'])

class Encoder():
    def __init__(self):
        model_weights_path = rospy.get_param("param_model_weights_path")
        
        # ROS
        rospy.init_node(name="encoder_node", anonymous=False)        
        self.subscriber = rospy.Subscriber(name="/carla/ego_vehicle/rgb_front/image",
                                            data_class=ImageMsg,
                                            callback=self.encoder_callback)
        self.publisher = rospy.Publisher(name="encoded_imgs/", data_class=EncodedImageVAE, queue_size=10)
        # CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu' # To run on a single workstation together with Carla
        # VAE model
        self.net = bmshj2018_hyperprior(quality=1, pretrained=False)
        self.net.load_state_dict(torch.load(model_weights_path))
        self.net.eval()
        self.net.to(self.device)
        # print(f'Parameters: {sum(p.numel() for p in self.net.parameters())}')

    def encoder_callback(self, in_img):
        global timing_df
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_compress = torch.cuda.Event(enable_timing=True)
        end_compress = torch.cuda.Event(enable_timing=True)

        
        start.record()
        timestamp_start = datetime.now().strftime("%H:%M:%S.%f")

        id = in_img.header.seq
        # type(ImageMsg.data)=uint8 in ROS
        # Encoding from carla = bgra -> delete 'a' channel and invert array -> result is 'rgb'-ordering
        img_data = np.frombuffer(in_img.data, dtype=np.uint8).reshape((in_img.height, in_img.width, 4))[:,:,:-1][:,:,::-1]
        img = Image.fromarray(img_data, 'RGB')
        # Create tensor, add dimension, resize
        tensor = transforms.ToTensor()(img.resize((256,256))).unsqueeze(0).to(self.device)

        start_compress.record()
        encoded_img = self.encode(tensor)
        end_compress.record()
        
        strings_ros = []
        for b_list in encoded_img['strings']:
            bytes_list_ros = []
            for byte in b_list:
                ints = []
                for i in byte:
                    ints.append(i)
                bytes_list_ros.append(IntList(ints))                
            strings_ros.append(BytesList(bytes_list_ros))
        
        
        out_encoded_img = EncodedImageVAE()
        h = Header(in_img.header.seq, rospy.Time.now(), in_img.header.frame_id)
        out_encoded_img.header = h 
        out_encoded_img.strings = strings_ros
        out_encoded_img.strings_shape = list(encoded_img['shape'])
        out_encoded_img.original_size = [(in_img.height), (in_img.width)]
        
        self.publisher.publish(out_encoded_img)
        
        
        end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) # milliseconds
        elapsed_time_compression = start_compress.elapsed_time(end_compress) # milliseconds
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")
        timing_df = timing_df.append({'frameID': id, 'inference_time': elapsed_time, 'inference_time_compression': elapsed_time_compression, 'timestamp_end': timestamp, 'timestamp': timestamp_start}, ignore_index=True)
        
        

    def encode(self, img_tensor):
        with torch.no_grad():
            # [JJ] type(output_encoder) = dict
            output_encoder = self.net.compress(img_tensor)
        return output_encoder

def write_timing():
    # global timing_df # maybe needed, try it out!    
    # timing_df.to_csv("SAVING_PATH")

if __name__ == "__main__":
    listener = Encoder()    
    # rospy.on_shutdown(write_timing)
    rospy.spin()
