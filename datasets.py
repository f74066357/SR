import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms
import glob
import re

class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name
        self.images=[]

        assert self.split in {'train', 'val','test'}
        
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"
        
        # Read list of image-paths
        if self.split == 'test':
            for images in sorted(glob.glob("./data/testing_lr_images*.png"),key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]):
                self.images.append(images)
            for dir_path, dir_names, file_names in os.walk('./data/testing_lr_images'):
                for f in file_names:
                    self.images.append(os.path.join(dir_path, f))
        else:
            for dir_path, dir_names, file_names in os.walk(os.path.join('./data', self.split)):
                for f in file_names:
                    self.images.append(os.path.join(dir_path, f))

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)