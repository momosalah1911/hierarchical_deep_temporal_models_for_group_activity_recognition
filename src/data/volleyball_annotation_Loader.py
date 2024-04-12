from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class Video_annotation_loader(Dataset):
    def __init__(self, src_folder, annotation_file, transformation=None):
        super().__init__()
        self.src_folder = src_folder
        self.annotation_file = pd.read_csv(annotation_file)
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.annotation_file.iloc[index][[
            'video', 'MainFrame']]
        image_path = f"{self.src_folder}\{video}\{MainFrame[:-4]}\{MainFrame}"
        print(image_path)
        image = plt.imread(image_path)
        y_label = self.annotation_file.iloc[index]['label']
        if self.transformation:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            image = self.transformation(image)
        return [image, y_label]


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(720, 1280)),
])
Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'
annotation_df = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.csv'
x = Video_annotation_loader(Video_folder_path, annotation_df, train_transform)
training_data = DataLoader(dataset=x, batch_size=1, shuffle=True)
out = next(iter(training_data))
title = out[1][0]
image_tensor = out[0][0]
image_np = image_tensor.permute(1, 2, 0).numpy()
plt.imshow(image_np)
plt.title(title)
plt.show()
