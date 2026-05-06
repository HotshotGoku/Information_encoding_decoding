from utils.config import SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from utils.preprocess import preprocess_simulation_graybackground, preprocess_seed

source_folder= SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS
target_folder= SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS

class MyDataset(Dataset):
    def __init__(self, source_folder:str=SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, target_folder:str=SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, start:int=0,end:int=30000, use_vae=False, preprocess_seed_output= True):
        # Store full paths to avoid repeated path joining
        # choose the first N files for consistency
        self.use_vae= use_vae
        self.source_files = sorted([os.path.join(source_folder, f) for f in os.listdir(source_folder)])[start:end]
        self.target_files = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder)])[start:end]
        self.preprocess_seed_output = preprocess_seed_output
        
        # Sanity check
        assert len(self.source_files) == len(self.target_files), \
            f"Mismatch: {len(self.source_files)} sources vs {len(self.target_files)} targets"


    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]

        # Load and preprocess the data 
        source = preprocess_simulation_graybackground(source_path)
        if self.use_vae== False:
            target = preprocess_seed(target_path)
        elif self.use_vae== True and self.preprocess_seed_output== True:
            target = preprocess_seed(target_path, left_crop=2, right_crop=3, img_length=32, img_width=32)  # VAE latent size
        else:
            target = preprocess_seed(target_path, img_length=32, img_width=32)  # no cropping if already preprocessed to 32x32


        # Normalize source and target images to [0,1] for Logits loss 

        source = source.astype('float32') / 255.0
        target = target.astype('float32') / 255.0

        # add channel dimension assuming grayscale and convert to torch tensors
        source = source[np.newaxis, :, :]  # shape (1, H, W)
        target = target[np.newaxis, :, :]  # shape (1, H, W)    

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        fname= os.path.basename(source_path)
        stem= os.path.splitext(fname)[0]

        return {'source': source, 'target': target, 'stem': stem}



       