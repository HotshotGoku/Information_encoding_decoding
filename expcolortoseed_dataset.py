
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from utils.preprocess import preprocess_experimental_backgroundwhite, preprocess_seed
import cv2
from natsort import natsorted


from utils.config import SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, SPECIFIC_FOLDER_SEED_DIFFUSION_MORE

source_folder= SPECIFIC_FOLDER_EXP_DIFFUSION_MORE
target_folder= SPECIFIC_FOLDER_SEED_DIFFUSION_MORE


class MyDataset(Dataset):
    def __init__(self, source_folder:str=SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, target_folder:str=SPECIFIC_FOLDER_SEED_DIFFUSION_MORE, start:int=0,end:int=None, use_vae=False, preprocess_method=cv2.INTER_NEAREST,preprocess_input="backgroundwhite"):
        # Store full paths to avoid repeated path joining
        # choose the first N files for consistency
        if end is None:
            end= len(os.listdir(source_folder))
        self.use_vae= use_vae   
        self.source_files = natsorted([os.path.join(source_folder, f) for f in os.listdir(source_folder)])[start:end]
        self.target_files = natsorted([os.path.join(target_folder, f) for f in os.listdir(target_folder)])[start:end]
        self.preprocess_method= preprocess_method
        self.preprocess_input= preprocess_input
        
        # Sanity check
        assert len(self.source_files) == len(self.target_files), \
            f"Mismatch: {len(self.source_files)} sources vs {len(self.target_files)} targets"


    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]

        # Load and preprocess the data 
        if self.preprocess_input == "backgroundwhite":
            source = preprocess_experimental_backgroundwhite(source_path) # keeping color input 
        else:
            # no cropping here
            source= cv2.imread(source_path, cv2.IMREAD_COLOR)
            source= cv2.cvtColor(source, cv2.COLOR_BGR2RGB)  # convert to RGB for consistency with previous preprocessing
            source= cv2.resize(source, (256,256)) # already 256x256, just resizing for consistency 

        if self.use_vae== False:
            target = preprocess_seed(target_path, method=self.preprocess_method)
        else:
            target = preprocess_seed(target_path, img_length=32, img_width=32)  # VAE latent size
            
        # Normalize source and target images to [0,1] for Logits loss 
        source = source.astype('float32') / 255.0
        target = target.astype('float32') / 255.0

        # since the source now is RGB image, no need to add channel dimension, but need to shuffle to (C,H,W)
        source = np.transpose(source, (2,0,1))  # shape (3, H, W)

        # add channel dimension assuming grayscale and convert to torch tensors
        target = target[np.newaxis, :, :]  # shape (1, H, W)    

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        fname= os.path.basename(source_path)
        stem= os.path.splitext(fname)[0]

        return {'source': source, 'target': target, 'stem': stem}



       