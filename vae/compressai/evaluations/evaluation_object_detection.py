import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import os

import cv2

model_type = "VAE" # "GAN", "ORIG"

def run_detector(detector, img):
  converted_img  = tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(img)[tf.newaxis, ...], dtype=tf.uint8)
  result = detector(converted_img)
  result = {key:value.numpy() for key,value in result.items()}

  return result

module_handle = "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1"
detector = hub.load(module_handle)

if model_type == "VAE":
  quality_levels = ["high" , "low" , "med_high",  "medium" , "med_low" , "super_low"]
elif model_type == "GAN": 
  quality_levels = ["high-super_high", "low", "low-med", "med"]
elif model_type == "ORIG":
  quality_levels = ["original"]
else:
  print("Undefined model type!")

for quality_level in quality_levels:
    if model_type == "VAE":
      images_path = #...
    elif model_type == "GAN": 
      images_path = #...
    elif model_type == "ORIG":
      images_path = #... 
    else:
      print("Undefined model type!")

    df = pd.DataFrame(columns = ["image_name", "detection_network", "model", "quality_level", "class", "score", "ymin", "xmin", "ymax", "xmax" ])
    count = 0
    total_images = len([x for x in os.listdir(images_path)])

    for image_file in os.listdir(images_path):
      if image_file.endswith(".png"):
        count = count + 1
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
        out_dict = run_detector(detector, image)
        detection_classes = out_dict["detection_classes"][0]
        detection_boxes = out_dict["detection_boxes"][0]
        detection_scores = out_dict["detection_scores"][0]

        for obj_class, obj_box, obj_score in zip(detection_classes, detection_boxes, detection_scores):
            if obj_score > 0.25:
                obj_class = obj_class
                ymin, xmin, ymax, xmax = obj_box

                obj_dict = {"image_name": image_file,
                            "detection_network": "ssd_mobilenet_v2",
                            "model": model_type,
                            "quality_level": quality_level,
                            "class": obj_class,
                            "score": obj_score,
                            "ymin": ymin,
                            "xmin": xmin,
                            "ymax": ymax,
                            "xmax": xmax}

                df = df.append(obj_dict, ignore_index=True)
                print(f"{count} / {total_images} | {quality_level}", end="\r")
        
    df.to_csv(f"./object_detection_results_{model_type}_{quality_level}.csv")