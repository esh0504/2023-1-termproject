import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms
import itertools
import random

class VeinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.finger_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.pairs = []
        
        # Calculate all possible pairs
        for folder in self.finger_folders:
            folder_path = os.path.join(self.root_dir, folder)
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))] # Save the full path of images
            self.pairs.extend(itertools.combinations(images, 2)) # all possible positive pairs in this folder

        self.len = len(self.pairs) * 2 # positive pairs and negative pairs are in 1:1 ratio

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Positive pair
        if idx < len(self.pairs):
            pair = self.pairs[idx]
            anchor_image = Image.open(pair[0])
            positive_image = Image.open(pair[1])
            if self.transform:
                anchor_image = self.transform(anchor_image)
                positive_image = self.transform(positive_image)
            return anchor_image, positive_image, torch.tensor(1.0, dtype=torch.float32)
        # Negative pair
        else:
            idx -= len(self.pairs)
            pair = self.pairs[idx]
            anchor_image = Image.open(pair[0])
            negative_finger_folder = random.choice([d for d in self.finger_folders if d != pair[0].split('/')[0]]) # select a different folder
            negative_finger_path = os.path.join(self.root_dir, negative_finger_folder)
            negative_images = [f for f in os.listdir(negative_finger_path) if os.path.isfile(os.path.join(negative_finger_path, f))]
            negative_image_name = random.choice(negative_images)
            negative_image = Image.open(os.path.join(negative_finger_path, negative_image_name))
            if self.transform:
                anchor_image = self.transform(anchor_image)
                negative_image = self.transform(negative_image)
            return anchor_image, negative_image, torch.tensor(0.0, dtype=torch.float32)