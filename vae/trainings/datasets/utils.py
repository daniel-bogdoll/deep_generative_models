# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None):#, split="train"):
        #splitdir = Path(root) / split
        data_dir = Path(root)

        if not data_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in data_dir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class PointCloudFolder(Dataset):
    
    def __init__(self, root, transform=None, cloud_arg="kitti_cloud_2d"):
        data_dir = Path(root)

        if not data_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in data_dir.iterdir() if f.is_file()]

        self.transform = transform
        self.cloud_arg = cloud_arg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            cloud2d: PointCloud in 2d representation
        """
        try:
            file = open(self.samples[index], 'rb')
            data = np.load(file)
        except:
            file = open(self.samples[index+1], 'rb')
            data = np.load(file)

        cloud_2d = data[self.cloud_arg][:,:,0:3]

        if self.transform:
            return self.transform(cloud_2d)
        return cloud_2d

    def __len__(self):
        return len(self.samples)