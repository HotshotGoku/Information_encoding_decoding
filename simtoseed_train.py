import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models import UNet
from simtoseed_dataset import MyDataset


# configs

batch_size = 128
learning_rate = 1e-5

model=UNet()
model.learning_rate = learning_rate

# Misc  
dataset = MyDataset()
dataloader= DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
trainer=   pl.Trainer(gpus=1, precision=32, max_epochs=500)

# Train!
trainer.fit(model, dataloader)



