import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2
import glob

from utils.config import SEED_FOLDER_TEST_INFOENCODING_MORESEEDS, SPECIFIC_FOLDER_SEED_DIFFUSION_MORE
from utils.preprocess import preprocess_seed

# Load the sorted files from the seed folder

SEED_DIR= SEED_FOLDER_TEST_INFOENCODING_MORESEEDS
OUTPUT_DIR= SPECIFIC_FOLDER_SEED_DIFFUSION_MORE
os.makedirs(OUTPUT_DIR, exist_ok=True)

seed_files= sorted(glob.glob(os.path.join(SEED_DIR, "*.png")))[:30000]

# Each file in the list will be first processed with preprocess seed
# Then we will create three triplicate copies(for the corresponding samples in diffusion model) and save it as x_1,x_2,x_3.png
# where x is the orginal file name without extension.
# new files will be saved in OUTPUT_DIR

for file_idx, seed_file in enumerate(seed_files):

    # Preprocess the seed image
    preprocessed_img = preprocess_seed(seed_file, left_crop=2, right_crop=3, img_length=32, img_width=32)

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(seed_file))[0]

    # Save three triplicate copies with modified names
    for i in range(1, 4):
        new_filename = f"{base_filename}_{i}.png"
        new_filepath = os.path.join(OUTPUT_DIR, new_filename)
        cv2.imwrite(new_filepath, preprocessed_img)
    
    if file_idx % 1000 == 0:  # Print progress every 1000 files
        print(f"Processed {seed_file} and saved triplicates as {new_filename}")
    
    
print("Processing completed. Preprocessed seed images with triplicates saved in the output directory.")