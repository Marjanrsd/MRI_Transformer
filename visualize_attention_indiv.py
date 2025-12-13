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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

IMAGE_SIZE = 182 
DISCARD_RATIO = 0.8 # how many image patches are disregarded during the attention process.

nii_file = "/mnt/chrastil/users/marjanrsd/vit/resized_T1/sub-1001_T1w_resized.nii.gz" # put this in a for loop. (182, 182, 182)
img = nib.load(nii_file)
img_data = img.get_fdata()

# Normalize to [0, 1]
img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
# Resize to (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
zoom_factors = [IMAGE_SIZE / s for s in img_data.shape]
img_resized = zoom(img_data, zoom=zoom_factors, order=1) # (182, 182, 182)
print("IMG_RESIZED", img_resized.shape)
# Convert to torch tensor: shape (1, 1, D, H, W)
input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

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
    num_classes=1,  # Adjust if you're using this model for regression
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


# Load the trained weights
checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/tl_model.pth"
checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/best_no_transfer_BS16_sigmoid.pth"
checkpoint_path = "/mnt/chrastil/users/marjanrsd/vit/best_models_indiv_tasks/best_task0.pth"
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

# ----------- Attention Rollout (no category index) -----------
resized_dir = "/mnt/chrastil/users/marjanrsd/vit/resized_T1"
subject_files = sorted([f for f in os.listdir(resized_dir) if f.endswith("_T1w_resized.nii.gz")])
subject_files = [subject_files[0]]
print(subject_files)
for subject_file in subject_files:
    subject_id = subject_file.split("_")[0]  # e.g., sub-1001
    nii_file = os.path.join(resized_dir, subject_file)
    print(f"\nProcessing {subject_id}...")
    img = nib.load(nii_file)
    img_data = img.get_fdata()

    # Normalize to [0, 1]
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    # Resize to (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    zoom_factors = [IMAGE_SIZE / s for s in img_data.shape]
    img_resized = zoom(img_data, zoom=zoom_factors, order=1)
    # Convert to torch tensor: shape (1, 1, D, H, W)
    input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    rollout_strategy = "attention_rollout"
    category_index = 0
    print(f"Using {rollout_strategy}")
    if rollout_strategy == "attention_rollout":
        rollout = VITAttentionRollout(model, discard_ratio=DISCARD_RATIO)
        mask = rollout(input_tensor)
        top_n = 4913
        flat_idx = np.argpartition(mask.flatten(), -top_n)[-top_n:]
        attn_coords = np.array(np.unravel_index(flat_idx, mask.shape)).T  # shape: (N, 3)
        np.save("1001_task0_attn_coords.npy", attn_coords)

        print(f"Top-{top_n} attention coords (attention space):\n", attn_coords)
    else:
        raise ValueError("Only attention_rollout is currently supported")


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
    # import pdb; pdb.set_trace()
     # Optional: Enhance contrast with a power-law (gamma) transform
    
    mask_norm[mask_norm < 0.3] = 0  # suppress background
    mask_norm = np.clip(mask_norm, 0, 1)
    print(f'np.std(mask_norm): {np.std(mask_norm)}')
    print(f'np.unique(mask_norm): {np.unique(mask_norm)}')
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)





# Define slice indices to visualize
#slice_indices = [5, 30, 55, 80, 105, 130, 155, 180]
slice_indices = [80]
# Create a figure with subplots for each plane
fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 15))

zoom_factors = [IMAGE_SIZE / s for s in mask.shape]
# Ensure axes is always 2D (even if there's only one row of slices)
if len(slice_indices) == 1:
    axes = np.expand_dims(axes, axis=0)  # or use np.atleast_2d(axes)

resized_mask = zoom(mask, zoom=zoom_factors, order=0)
resized_mask = resized_mask / np.max(resized_mask)  # Normalize globally

## @TODO ensure these are in the right order e.g. [sag, cor, axial]
# Visualize sagittal slices
for i, slice_idx in enumerate(slice_indices):
    img_slice = img_resized[:,slice_idx, :]
    resized_mask = zoom(mask, zoom=zoom_factors, order=0) 
    mask_slice = resized_mask[:, slice_idx,:]
    overlay = show_mask_on_image(img_slice, mask_slice)
    axes[i, 0].imshow(overlay)
    axes[i, 0].set_title(f'Sagittal Slice {slice_idx}')
    axes[i, 0].axis('off')
    
    
 
'''
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
    im = axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f'Axial Slice {slice_idx}')
    axes[i, 2].axis('off')
'''
#fig.colorbar(im, ax=axes[:, 0], fraction=0.046, pad=0.04)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap='jet', norm=norm)
cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()



