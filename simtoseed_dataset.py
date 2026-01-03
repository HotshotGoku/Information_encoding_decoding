from utils.config import SEED_FOLDER_SEEDTOSIM, SIM_FOLDER_SEEDTOSIM
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from utils.preprocess import preprocess_simulation_graybackground, preprocess_seed

source_folder= SIM_FOLDER_SEEDTOSIM
target_folder= SEED_FOLDER_SEEDTOSIM

class MyDataset(Dataset):
    def __init__(self, source_folder:str=SIM_FOLDER_SEEDTOSIM, target_folder:str=SEED_FOLDER_SEEDTOSIM, start:int=0,end:int=30000):
        # Store full paths to avoid repeated path joining
        # choose the first N files for consistency
        self.source_files = sorted([os.path.join(source_folder, f) for f in os.listdir(source_folder)])[start:end]
        self.target_files = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder)])[start:end]
        
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
        target = preprocess_seed(target_path)

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



       