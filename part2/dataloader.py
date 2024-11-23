import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, GaussianBlur, Lambda, ColorJitter, Compose
import torchvision.transforms.functional as tvtf
from tvtf import to_tensor, to_pil_image
from PIL import Image
import os
from dataaug import *
from loadParam import *
import pdb

class WindowDataset(Dataset):
    def __init__(self, ds_path, transform=None, device='cpu'):
        # init code
        self.input_data_path = ds_path + "/Input_Data/"
        self.labels_path = ds_path + "/Labels/"
        self.instance_labels_path = ds_path + "/Instance_Labels/"
        self.transform = transform

        print("Dataset Initialized")
        print(f"Input Data: {self.input_data_path}")
        print(f"Ground Truth Labels: {self.labels_path}")
        print(f"Instance Segmentation Labels: {self.instance_labels_path}")

    def __len__(self):
        # Set the dataset size here
        N = len([name for name in os.listdir(self.input_data_path) if os.path.isfile(os.path.join(self.input_data_path, name))])
        return N

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        
        # Open the RGB image and ground truth label
        input_path = self.input_data_path + f"Image{idx:04d}.jpg"
        labels_path = self.labels_path + f"Image{idx:04d}.jpg"
        instance_labels_path = self.instance_labels_path + f"Image{idx:04d}.exr"

        # convert them to tensors
        input_tensor = self.transform(input_path)
        labels_tensor = self.transform(labels_path)
        instance_labels_tensor = self.transform(instance_labels_path)

        # apply any transform (blur, noise...)
        # blur_transform = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        # noise_transform = Lambda(lambda img: img + torch.randn_like(img) * 0.1)
        # jitter_transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        
        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    DS_PATH = "C:/Users/pdbru/Documents/DL_for_Perception/rbe474x_p2/Dataset"
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)
    print(dataloader.__len__())

    rgb, label = dataset[0]
