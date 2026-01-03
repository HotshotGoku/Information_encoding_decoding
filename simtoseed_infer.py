from utils.config import CKPT_PATH_SEEDTOSIM, OUTPUT_DIR_SEEDTOSIM
from simtoseed_dataset import MyDataset
import os 
from models import UNet
from torch.utils.data import DataLoader
import torch 
import cv2

# test set
from utils.config import SEED_FOLDER_SEEDTOSIM_TEST, SIM_FOLDER_SEEDTOSIM_TEST

model = UNet.load_from_checkpoint(checkpoint_path=CKPT_PATH_SEEDTOSIM)
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Dataloader

# dataset= MyDataset(start=50000, end=50100) # infer on 100 images not in training set
dataset= MyDataset(source_folder=SIM_FOLDER_SEEDTOSIM_TEST, target_folder=SEED_FOLDER_SEEDTOSIM_TEST,start=0,end=7)
dataloader= DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# Inference loop

os.makedirs(OUTPUT_DIR_SEEDTOSIM, exist_ok=True)

with torch.no_grad():
    for i,batch in enumerate(dataloader):
        source = batch['source'].to(device)  # shape (B, 1, H, W)
        
        logits = model(source)  # shape (B, 1, H, W)
        # convert logits to probabilities
        prob=torch.sigmoid(logits)

        # threshold at 0.5 for binary mask
        # binary_mask= (prob > 0.5).float()
        # output = binary_mask.cpu().numpy()  # convert to numpy array

        output = prob.cpu().numpy()  # convert to numpy array

        # Save each image in the batch
        for j in range(output.shape[0]):
            img_array = (output[j, 0, :, :] * 255).astype('uint8')  # denormalize and convert to uint8
            save_path = os.path.join(OUTPUT_DIR_SEEDTOSIM, f"{batch['stem'][j]}.png")
            cv2.imwrite(save_path, img_array)