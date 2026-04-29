import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from models import UNet_Resblocks
from Information_encoding_decoding.models.extra.ResNet_old import LatentToIC
# from models.ResNet_dilated import PDEArenaDilatedResNet
from expcolortoseed_dataset import MyDataset
from utils.config import SPECIFIC_FOLDER_SEED

# configs 

batch_size = 128
learning_rate = 5e-4
lambda_= 0.0
pos_weight= None
dropout = 0.3  # Add dropout for regularization
weight_decay = 1e-4  # L2 regularization
name=f"ResNet_exps_color_vae_nonrandom_v3"
# version= 'dilated'
version = 'old'

# model=UNet_Resblocks(in_channels=4,use_vae=True, features=[16,32,64], learning_rate=learning_rate, lambda_=lambda_, pos_weight=pos_weight, dropout=dropout, weight_decay=weight_decay)
model=LatentToIC(use_vae=True, learning_rate=learning_rate, lambda_=lambda_, weight_decay=weight_decay)
# model = PDEArenaDilatedResNet(use_vae=True, learning_rate=learning_rate)
# Misc  
dataset = MyDataset(use_vae=True, target_folder=SPECIFIC_FOLDER_SEED)

# Fix split to avoid augmentation leakage
# 409 base images × 100 augmentations = 40,900 total
# Take 41 base images for validation (clean 4,100 images)
n_augmentations_per_image = 100
n_val_base_images = 41
n_train = (409 - n_val_base_images) * n_augmentations_per_image  # 36,800
n_val = n_val_base_images * n_augmentations_per_image  # 4,100

# make non random splits to prevent data leakage from augmentation versions
train_indices= list(range(0, n_train))
val_indices= list(range(n_train, n_train+n_val))

train_ds,val_ds= torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

# train_ds, val_ds= random_split(dataset, [n_train, n_val], generator= torch.Generator().manual_seed(42))


train_loader= DataLoader(train_ds, num_workers=8, batch_size=batch_size, shuffle=True, persistent_workers=True,pin_memory=True)
val_loader= DataLoader(val_ds, num_workers=8, batch_size=batch_size, shuffle=False, persistent_workers=True,pin_memory=True)
loggers = [TensorBoardLogger("lightning_logs", name=name, version=version), CSVLogger("lightning_logs", name=name, version=version)]

callbacks = [
    ModelCheckpoint(dirpath=f"lightning_logs/{name}/{version}/checkpoints", monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best-{epoch:03d}-{val_loss:.4f}"),
    EarlyStopping(monitor="val_loss", mode="min", patience=50),
    LearningRateMonitor(logging_interval="epoch"),
]

trainer=   pl.Trainer(accelerator="gpu", devices=1, precision=32, max_epochs=500, logger=loggers, callbacks=callbacks)
# Train!
trainer.fit(model, train_loader, val_loader)


