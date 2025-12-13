import os
import sys
import cv2
import torch
import requests
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchio as tio
import nibabel as nib
from io import BytesIO
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torchvision import transforms
from vit_pytorch.cct_3d import CCT
from collections import OrderedDict
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from collections import Counter

def parse_schaefer_lut(lut_path):
    lut_dict = {}
    with open(lut_path, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.strip().split()
            try:
                idx = int(parts[0])
                name = parts[1]
                lut_dict[idx] = name
            except (ValueError, IndexError):
                continue
    return lut_dict


block_size_x = 240/17
block_size_y = 320/17
block_size_z = 320/17

def attn_voxels_to_labels(attn_coords, parc_data):
    #labels = []
    unique_voxels = set()
    #voxel_coords = []
    for (x, y, z) in attn_coords:
        x_start = max(0, int(np.floor(x * block_size_x)))
        y_start = max(0, int(np.floor(y * block_size_y)))
        z_start = max(0, int(np.floor(z * block_size_z)))
        
        x_end = min(parc_data.shape[0], int(np.ceil((x + 1) * block_size_x)))
        y_end = min(parc_data.shape[1], int(np.ceil((y + 1) * block_size_y)))
        z_end = min(parc_data.shape[2], int(np.ceil((z + 1) * block_size_z)))
        
        # Get unique voxel coordinates
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                for k in range(z_start, z_end):
                    unique_voxels.add((i, j, k))
    
    labels = []
    for (i, j, k) in unique_voxels:
        if 0 <= i < parc_data.shape[0] and 0 <= j < parc_data.shape[1] and 0 <= k < parc_data.shape[2]:
            label = parc_data[i, j, k]
            if label != 0:  # Only include non-zero labels
                labels.append(int(label))
    return labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

IMAGE_SIZE = 182 
DISCARD_RATIO = 0.0 # how many image patches are disregarded during the attention process.

# ----------- Load your trained 3D ViT model -----------
model = CCT(
    img_size=IMAGE_SIZE,
    num_frames=IMAGE_SIZE,
    embedding_dim=126,  # adjust based on your model
    n_conv_layers=4,
    n_input_channels=1,
    frame_kernel_size=3,
    kernel_size=2,
    stride=1,
    frame_stride=1,
    padding=3,
    frame_padding=3,
    pooling_kernel_size=2,
    frame_pooling_kernel_size=2,
    pooling_stride=2,
    frame_pooling_stride=2,
    pooling_padding=1,
    frame_pooling_padding=1,
    num_layers=2,
    num_heads=3,
    mlp_ratio=1.0,
    num_classes=1,  # 68 if you're using total tasks model. Adjust if you're using this model for regression
    positional_embedding='sine'
).to(DEVICE)



# Add the updated FC layer (from training)
model.transformer.fc = nn.Sequential(
    nn.Linear(model.transformer.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
    nn.Sigmoid()
).to(DEVICE)

'''
# Load the trained weights
checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/tl_model.pth"
checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/best_no_transfer_BS16_sigmoid.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# If it's a wrapped dict (e.g., saved with {'state_dict': ...}), unwrap it
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove 'module.' from keys if present (used when saving the model with multiple GPUs)
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

# Load the cleaned state dict into the model
model.load_state_dict(new_state_dict)
model.eval()
model.to(DEVICE)
'''
# ----------- Attention Rollout (no category index) -----------
resized_dir = r"/mnt/chrastil/users/marjanrsd/vit/resized_T1"
base_output_dir = r"/mnt/chrastil/users/marjanrsd/vit/mri_transformer/schaefer_attended_regions_each_task"

task_model_paths = [
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task0.pth",
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task1.pth",
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task2.pth",
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task3.pth",
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task4.pth",
    "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task5.pth"
    ]

total_voxels_df = pd.read_csv(r"/mnt/chrastil/users/marjanrsd/vit/mri_transformer/dbg3_schaefer_attended_regions/total_voxels.csv")
total_voxels_df.columns = ["Region", "TotalVoxels"]
lut_path = "/mnt/chrastil/users/marjanrsd/ants/atlas/Schaefer2018_400Parcels_7Networks_order.txt"
region_names = parse_schaefer_lut(lut_path)
''' if you want to join regions with the same name
for (k,v) in region_names.items():
    region_names[k] = '_'.join(v.split('_')[:-1])
''' 
region_ints = {v:k for k, v in region_names.items()}
subject_files = sorted([f for f in os.listdir(resized_dir) if f.endswith("_T1w_resized.nii.gz")])
#subject_files = subject_files[0:2] # @TODO rm
for task_index, model_path in enumerate(task_model_paths):
    if task_index in {0, 1, 2}:
        continue
    
    task_output_dir = os.path.join(base_output_dir, f"task{task_index}")
    os.makedirs(task_output_dir, exist_ok=True)
    print(f"\nLoading model for task {task_index}")

    # Load the trained weights
    #checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/best_no_transfer_BS16_sigmoid.pth"
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Load the cleaned state dict into the model
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(DEVICE)

    #subject_files = ["sub-1001_T1w_resized.nii.gz"]
    
    for subject_file in subject_files:
        subject_id = subject_file.split("_")[0].replace("sub-", "")  # e.g., sub-1001
        nii_file = os.path.join(resized_dir, subject_file)
        print(f"\nProcessing {subject_id}...")
        img = nib.load(nii_file)
        img_data = img.get_fdata()
        # Normalize to [0, 1]
        # @TODO ensure this is the same normalization used in train.py
        # img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        # Resize to (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        zoom_factors = [IMAGE_SIZE / s for s in img_data.shape]
        img_resized = zoom(img_data, zoom=zoom_factors, order=1) # (182, 182, 182)
        # Convert to torch tensor: shape (1, 1, D, H, W)
        input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        rollout = VITAttentionRollout(model, discard_ratio=DISCARD_RATIO)
        mask = rollout(input_tensor) # (17, 17, 17)
        top_n = 50 #100
        schaefer_path = f"/mnt/chrastil/users/marjanrsd/vit/mri_transformer/schaefer_results/{subject_id}Schaefer_in_subject_space.nii.gz"
        schaefer_data = nib.load(schaefer_path).get_fdata()
        # Flatten the attention mask and get sorted indices (from highest to lowest)
        flat_mask = mask.flatten()
        sorted_indices = np.argsort(flat_mask)[::-1]  # descending order
        all_coords = np.array(np.unravel_index(sorted_indices, mask.shape)).T  # list of (x, y, z)
        collected_regions = Counter()  # To store cumulative ROI hits

        for coord in all_coords: # iterate over all 3D tokens
            print("starting coordinates")
            attn_labels = attn_voxels_to_labels([tuple(coord)], schaefer_data) # pass it as a list
            filtered_labels = [int(label) for label in attn_labels if int(label) != 0]
            
            region_counts = Counter([region_names.get(int(label), 'Unknown') for label in filtered_labels])
            collected_regions.update(region_counts)

            if len(collected_regions) >= top_n:
                print(f"got the fisrt {top_n} ROIs!")
                break
        
        # normalize wrt each subject's total brain region size
        for k, v in collected_regions.items():
            collected_regions[k] = v / np.sum(schaefer_data == region_ints[k])

        df = pd.DataFrame(collected_regions.items(), columns=["Region", "ProportionAttended"])
        #df = df.merge(total_voxels_df, on="Region", how="left")
        #df["ProportionAttended"] = df["Count"] / df["TotalVoxels"]
        
        output_csv = os.path.join(task_output_dir, f"{subject_id}_task{task_index}_schaefer_attended_regions.csv")
        df.to_csv(output_csv, index=False)
        print(f"saved {subject_file} to {output_csv} file")


# ----------- Visualization (2D axial slice) -----------
def show_mask_on_image(img_slice, mask_slice):
    img = np.float32(img_slice)
    img = np.stack([img] * 3, axis=-1)
    mask_norm = mask_slice # - mask_slice.min()
    if mask_norm.max() != 0:
        mask_norm /= mask_norm.max()
    else:
        assert False
        mask_norm = np.zeros_like(mask_norm)
    random_mask = torch.rand_like(torch.tensor(mask_slice)).numpy()
    #heatmap = cv2.applyColorMap(np.uint8(255 * random_mask), cv2.COLORMAP_JET)
    # import pdb; pdb.set_trace()
    print(f'np.std(mask_norm): {np.std(mask_norm)}')
    print(f'np.unique(mask_norm): {np.unique(mask_norm)}')
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)

# Define slice indices to visualize
#slice_indices = [5, 30, 55, 80, 105, 130, 155, 180]
slice_indices = [80, 105]
# Create a figure with subplots for each plane
#fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 15))
num_slices = len(slice_indices)
fig, axes = plt.subplots(num_slices, 1, figsize = (5*num_slices, 5))
if num_slices == 1:
    axes = [axes]

zoom_factors = [IMAGE_SIZE / s for s in mask.shape]

## @TODO ensure these are in the right order e.g. [sag, cor, axial]
# Visualize sagittal slices
for i, slice_idx in enumerate(slice_indices):
    img_slice = img_resized[slice_idx, :, :]
    resized_mask = zoom(mask, zoom=zoom_factors, order=0) # for smoother results use order=3 - (182, 182, 182)
    mask_slice = resized_mask[slice_idx, :, :]
    overlay = show_mask_on_image(img_slice, mask_slice)
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f'Sagittal Slice {slice_idx}')
    axes[i, 1].axis('off')

    #mask_slice = mask[10, :, :]
    #mask_slice_int = ((mask_slice / np.max(mask_slice)) * 255).astype(np.uint8)
    #pil_image = Image.fromarray(mask_slice_int)
    #pil_image.save(f"og_mask_slice_{slice_idx}.png")

    
# Visualize coronal slices
for i, slice_idx in enumerate(slice_indices):
    img_slice = img_resized[:, slice_idx, :]
    resized_mask = zoom(mask, zoom=zoom_factors, order=0) 
    mask_slice = resized_mask[:, slice_idx, :]
    overlay = show_mask_on_image(img_slice, mask_slice)
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f'Coronal Slice {slice_idx}')
    axes[i, 1].axis('off')

# Visualize axial slices
for i, slice_idx in enumerate(slice_indices):
    img_slice = img_resized[:, :, slice_idx]
    resized_mask = zoom(mask, zoom=zoom_factors, order=0) 
    mask_slice = resized_mask[:, :, slice_idx]
    overlay = show_mask_on_image(img_slice, mask_slice)
    im = axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f'Axial Slice {slice_idx}')
    axes[i, 1].axis('off')
#fig.colorbar(im, ax=axes[:, 0], fraction=0.046, pad=0.04)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
norm = Normalize(vmin=0, vmax=1)


sm = ScalarMappable(cmap='jet', norm=norm)
cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()



'''
# rob's notes

region_names = {int: 'string'} # mapping
region_names[42] = my_fave_ROI_12 
region_names[43] = my_face_ROI_13
...

# now instead create region_names dict like this:
# everything before the number
base_ROI_str = full_str.split('_')[:-1]
# now add this ^^^

for (k,v) in region_names.item():
    region_names[k] = v.split('_')[:-1]

'''