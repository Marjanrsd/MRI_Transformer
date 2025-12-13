import csv
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

#@TODO print fast weights see if there's only 0s and 1s after learning

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
            self.data = self.data[self.data.iloc[:, 1] != -1]
            print("LENGTH DATA AFTER FILTERING", len(self.data))
        self.split = split
        self.deg = deg # data aug rotation amount


    def __len__(self):
        return len(self.data)
        #return 1 # for debugging only
        

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

        return voxel_tensor.squeeze(), label

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for voxels, labels in loader:
        voxels, labels = voxels.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(voxels.unsqueeze(1)) # Add channel dimension
        # NA values were replaced by -1
        labels[labels == -1] = outputs[labels == -1]
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Testing function
def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(voxels.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

num_tasks = [4]
#num_tasks = 6

if __name__ == "__main__":
    #for task in range(num_tasks):
    for task in num_tasks:
        print("TASK", task)
        # Load datasets
        train_data = MRIDataset(split="train", task=task)
        test_data = MRIDataset(split="test", task=task)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        # Training loop
        num_epochs = 25 #240
        best_loss = float("inf") # Initialize best loss as infinity
        # file to save the best test loss model thus far
        #load_path = "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task4.pth"
        save_path = f"/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task{task}.pth"
       
        try:
            checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint)
        except:
            pass

        train_losses = []
        test_losses = []

        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for epoch in range(num_epochs):
            test_loss = test_epoch(model, test_loader, criterion, device)
            #print(test_loss)
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            test_loss = test_epoch(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            print(f"task {task} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Save model if test loss improves
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved Best Model for task {task}) (Epoch {epoch+1})")

                with open(f"train_loss_task{task}_{dt_str}.csv", 'w', newline='') as csv_file:
                    wr = csv.writer(csv_file)
                    wr.writerow(train_losses)

                with open(f"test_loss_task{task}_{dt_str}.csv", 'w', newline='') as csv_file:
                    wr = csv.writer(csv_file)
                    wr.writerow(test_losses)
