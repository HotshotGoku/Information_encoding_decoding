# implement from the previous U-Net model, change to pytorch lightning 
import torch
import torch.nn as nn
import pytorch_lightning as pl

# trying to build one from scratch

class UNet(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]): # features does not include bottleneck, assumes each index is double of previous
        super().__init__()
        self.encoder= nn.ModuleList()
        self.decoder= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels= feature

        self.bottleneck= self.double_conv(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.double_conv(feature*2, feature))
        
        self.final_conv= nn.Conv2d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):

        skip_connections =[]

        for layer in self.encoder:
            x= layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x= self.bottleneck(x)

        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                x= layer(x)
                skip_connection= skip_connections.pop()
                if x.shape[2:] != skip_connection.shape[2:]:
                    x= torch.nn.functional.interpolate(x, size= skip_connection.shape[2:])
                    print(f"Error in combining skip connection, interpolated to {skip_connection.shape[2:]}")
                x= torch.cat((skip_connection, x), dim=1)
            else:
                x= layer(x)

        x= self.final_conv(x)
        return x

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr= self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x= train_batch['source']
        y=train_batch['target']
        y_hat= self(x)
        loss=nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x= val_batch['source']
        y= val_batch['target']
        y_hat= self(x)
        loss= nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('val_loss', loss)
    

    

