import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from models import UNet_Resblocks
from simtoseed_dataset import MyDataset
import os


from utils.config import SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS

# configs 
# run 1: lambda = 0.5, pos_weight = 2.0 


batch_size = 32
learning_rate = 5e-4
dropout = 0.3  # Add dropout for regularization
weight_decay = 1e-4  # L2 regularization
# Candidate hyperparameter values for grid search
lambda_ = 0.5
pos_weight_value = 2.0
pos_weight = torch.tensor([pos_weight_value])

name = "Res_UNet_sim_vae_nonrandom"
version = f"lambda{lambda_}_pos{pos_weight_value}_GPUsv3RotReps"
model=UNet_Resblocks(in_channels=4, use_vae=True, features=[16,32,64], learning_rate=learning_rate, lambda_=lambda_, pos_weight=pos_weight, dropout=dropout, weight_decay=weight_decay)
# model=LatentToIC(use_vae=True, learning_rate=learning_rate)
# model = PDEArenaDilatedResNet(use_vae=True, learning_rate=learning_rate)
# Misc  
# dataset = MyDataset(use_vae=True, target_folder=SPECIFIC_FOLDER_SEED_DIFFUSION, preprocess_input= 'None')
# dataset = MyDataset(use_vae=True, source_folder=SPECIFIC_FOLDER_EXP_DIFFUSION_ROTATED, target_folder=SPECIFIC_FOLDER_SEED, preprocess_input= 'None')
dataset = MyDataset(use_vae=True, source_folder=SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, target_folder=SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, preprocess_seed_output=False, start=0, end= 90000)

# Fix split to avoid augmentation leakage
# 409 base images × 100 augmentations = 40,900 total
# Take 41 base images for validation (clean 4,100 images)


# Now we have 30k images * 3 augs per image=90 k total
# 9000 images for validation, 81k for training
n_total= 30000  #409
n_augmentations_per_image =  3  #100
n_val_base_images = 3000     #41
n_train = (n_total - n_val_base_images) * n_augmentations_per_image  # 81k   # 36,800
n_val = n_val_base_images * n_augmentations_per_image # 9k # 4,100

# make non random splits to prevent data leakage from augmentation versions
train_indices= list(range(0, n_train))
val_indices= list(range(n_train, n_train+n_val))

train_ds,val_ds= torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

# train_ds, val_ds= random_split(dataset, [n_train, n_val], generator= torch.Generator().manual_seed(42))

num_gpus = 4
num_workers = max(1, (os.cpu_count() or 1) // num_gpus)  # CPUs per DDP process
train_loader= DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, persistent_workers=True,pin_memory=True)
val_loader= DataLoader(val_ds, num_workers=num_workers, batch_size=batch_size, shuffle=False, persistent_workers=True,pin_memory=True)
loggers = [TensorBoardLogger("lightning_logs", name=name, version=version), CSVLogger("lightning_logs", name=name, version=version)]

callbacks = [
    ModelCheckpoint(dirpath=f"lightning_logs/{name}/{version}/checkpoints", monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best-{epoch:03d}-{val_loss:.4f}"),
    EarlyStopping(monitor="val_loss", mode="min", patience=50),
    LearningRateMonitor(logging_interval="epoch"),
]

trainer=   pl.Trainer(accelerator="gpu", devices=4, strategy="ddp", precision=32, max_epochs=500, logger=loggers, callbacks=callbacks)
# Train!
trainer.fit(model, train_loader, val_loader)




