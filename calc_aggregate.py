import nibabel as nib
import pandas as pd
import numpy as np
import os
from collections import Counter
import csv
import os


# Load Schaefer 400 parcellation file
schaefer_400_file = "/mnt/chrastil/users/marjanrsd/ants/atlas/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
schaefer_img = nib.load(schaefer_400_file)
base_attn_dir = "/mnt/chrastil/users/marjanrsd/vit/mri_transformer/attention_coords_each_task/"
base_output_dir = "/mnt/chrastil/users/marjanrsd/vit/mri_transformer/schaefer_attended_regions_each_task/"

# Get the data (this is a 3D array)
schaefer_data = schaefer_img.get_fdata()
total_voxels_df = pd.read_csv(r"/mnt/chrastil/users/marjanrsd/vit/mri_transformer/dbg3_schaefer_attended_regions/total_voxels.csv")
total_voxels_df.columns = ["Region", "TotalVoxels"]
print(total_voxels_df.head)


for task in range(6): 
    #if task in {0, 1, 2}:
        #continue
    print(f"\n>>> Aggregating Task {task} Results...")
    roi_counter = {}
    roi_sum = {}
    roi_max = {}  # Track maximum proportion for each ROI

    csv_folder = os.path.join(base_output_dir, f"task{task}")
    num_subjects = len([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    print("number of subjects:",num_subjects)
    for filename in os.listdir(csv_folder):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(csv_folder, filename)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if 'ProportionAttended' not in fieldnames or 'Region' not in fieldnames:
                print(f"Skipping {filename} - required columns missing")
                continue

            for row in reader:
                roi = row['Region']
                voxel_value = float(row['ProportionAttended'])
                
                # Update sum and counter
                roi_sum[roi] = roi_sum.get(roi, 0.0) + voxel_value
                roi_counter[roi] = roi_counter.get(roi, 0) + 1
                
                # Track maximum proportion
                #if roi not in roi_max or voxel_value > roi_max[roi]:
                    #roi_max[roi] = voxel_value

    # Compute average and ensure values stay between 0 and 1
    roi_avg = {roi: min(1.0, roi_sum[roi] / num_subjects) for roi in roi_sum}
    
    # Save to CSV
    df_agg = pd.DataFrame(list(roi_avg.items()), columns=["Region", "AverageProportionAttended"])
    df_agg = df_agg.sort_values(by="AverageProportionAttended", ascending=False)

    # Add maximum proportion column for reference
    #df_agg['MaxProportion'] = [roi_max[roi] for roi in df_agg['Region']]

    output_file = os.path.join(csv_folder, f"aggregate_average_proportions_task{task}.csv")
    df_agg.to_csv(output_file, index=False)
    print(f"Saved aggregated results for task {task} to:\n{output_file}")
    
    # Print some statistics
    print(f"\nStatistics for task {task}:")
    print(f"Number of ROIs: {len(roi_avg)}")
    print(f"Average proportion range: {min(roi_avg.values()):.4f} - {max(roi_avg.values()):.4f}")
    #print(f"Maximum proportion range: {min(roi_max.values()):.4f} - {max(roi_max.values()):.4f}")


    # Filter ROIs with average proportion attended > 0.2
    threshold = 0.002
    roi_above_threshold = {roi: avg for roi, avg in roi_avg.items() if avg >= threshold}

    df_threshold = pd.DataFrame(list(roi_above_threshold.items()), columns=["Region", "AverageProportionAttended"])
    #df_threshold['MaxProportion'] = [roi_max[roi] for roi in df_threshold['Region']]

    threshold_output_file = os.path.join(csv_folder, f"above_threshold_regions_task{task}.csv")
    df_threshold.to_csv(threshold_output_file, index=False)
    print(f"Saved thresholded results (> {threshold}) for task {task} to:\n{threshold_output_file}")

