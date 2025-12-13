import csv
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchio as tio
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from vit_pytorch.cct_3d import CCT
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                        module="torchio.data.image")
warnings.filterwarnings("ignore", category=FutureWarning, 
                        message=".*Series.__getitem__ treating keys as positions.*")

class MRIDataset(Dataset):
    def __init__(self, split='train', deg=5, task=None):
        # tasks should be None (all tasks), or 0 to 5 (mrt, pft, etc)
        # Load train.csv or test.csv
        self.data = pd.read_csv(f"/mnt/chrastil/users/marjanrsd/vit/{split}.csv")
        print("LENGTH OF DATA", len(self.data))
        if task != None:
            self.data = self.data.iloc[:, [0, task+1]]
            print("LENGTH DATA BEFORE FILTERING", len(self.data))
            #self.data = self.data[self.data.iloc[:, 1] != -1]
            print("LENGTH DATA AFTER FILTERING", len(self.data))
        self.split = split
        self.deg = deg # data aug rotation amount
        self.task = task

    def __len__(self):
        return len(self.data)
        
    def set_deg(self, d=0):
        self.deg = d

    def __getitem__(self, idx):
        # Get the path to the numpy files
        npy_path = self.data.iloc[idx, 0] # t1_path column (first col)
        voxel_data = np.load(npy_path) # Load 3D MRI scan

        # Get the labels (behavioral task scores)
        label = torch.tensor(self.data.iloc[idx, 1:], dtype=torch.float32)
        # label = label / 100 # this is a hyperparamter. Lazy man's normalization/scaling

        # Convert voxel data to tensor (float32 for neural networks)
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32).unsqueeze(0)

        if self.split == "train":
            # Rotates up to ±deg degrees
            random_rotation = tio.RandomAffine(scales=0., degrees=self.deg)
            rotated_T1 = random_rotation(voxel_tensor)
            # set negative values to 0
            rotated_T1[rotated_T1 < 0] = 0.
            voxel_tensor = rotated_T1

        return voxel_tensor.squeeze(), label, npy_path 

if __name__ == "__main__":

    num_tasks = 6
    for task in range(num_tasks):
        print(f"Task #: {task}")
        # Load datasets
        train_data = MRIDataset(split="train", task=task)
        test_data = MRIDataset(split="test", task=task)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

        # Model, loss, optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CCT(
        img_size = 182,
            num_frames = 182,
            embedding_dim = 126, # needs to be divisible by 6 for pos_emb
            n_conv_layers = 4,
            n_input_channels=1,
            frame_kernel_size = 3,
            kernel_size = 2,
            stride = 1,
            frame_stride = 1,
            padding = 3,
            frame_padding = 3,
            pooling_kernel_size = 2,
            frame_pooling_kernel_size = 2,
            pooling_stride = 2,
            frame_pooling_stride = 2,
            pooling_padding = 1,
            frame_pooling_padding = 1,
            num_layers = 2, # default was 14
            num_heads = 3, # parallel attn fxs. Same input can go thru different W_k & W_q's.
            mlp_ratio = 1., # how many neurons in a Transformer block's FC layers. Bigger = more neurons. 
            num_classes = 1, # we're going to regress & use MSE loss
            positional_embedding = 'sine').to(device)
        
        model.transformer.fc = nn.Sequential(
            nn.Linear(model.transformer.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ).to(device)
            
        model = model.to(device)
        load_path = f"/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task{task}.pth"
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        npy_dir = f"frozen_dataset/task{task}"
        os.makedirs(npy_dir, exist_ok=True)
        input("READ @TODO. we modified the code. Enter continue")
        
        model.eval()
        with torch.no_grad():
            # get training data
            for voxels, labels, npy_paths in train_loader:
                voxels, labels = voxels.to(device), labels.to(device)
                #outputs = model(voxels.unsqueeze(0))[1] # Add channel dimension
                # @TODO, change the outputs = model(voxels.unsqueeze(0))[0] to outputs = model(voxels.unsqueeze(0))[1]
                outputs = model(voxels.unsqueeze(0))[0]
                

                for i in range(outputs.shape[0]): # batch size = 1
                    x = outputs[i].detach().cpu().numpy()
                    subject_bit = npy_paths[i].split('/')[-1]
                    save_path =  f'{npy_dir}/{subject_bit}'
                    #np.save(save_path, x)
                    #print(f"X: {x}")

            # get testing data
            for voxels, labels, npy_paths in test_loader:
                voxels, labels = voxels.to(device), labels.to(device)
                outputs = model(voxels.unsqueeze(0))[1] 
                for i in range(outputs.shape[0]):
                    x = outputs[i].detach().cpu().numpy()
                    subject_bit = npy_paths[i].split('/')[-1]
                    save_path =  f'{npy_dir}/{subject_bit}'
                    #np.save(save_path, x)
                    print(f"X: {x}")
                