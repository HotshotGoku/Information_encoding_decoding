# implement from the previous U-Net model, change to pytorch lightning 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.vae import vae, encode_img
from losses.dice_loss import dice_loss


# trying to build one from scratch

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, dropout=0.0):
        super().__init__()
        self.channels_in= channels_in
        self.channels_out= channels_out
        self.relu= nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        if channels_in != channels_out:
            self.proj= nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.proj= None

        self.conv1= nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

        self.conv2= nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out)
        )

    def forward(self, x):
        if self.proj is None:
            identity= x
        else:
            identity= self.proj(x)

        out= self.conv1(x)
        if self.dropout is not None:
            out = self.dropout(out)
        out= self.conv2(out)
        
        out+= identity
        out= self.relu(out)
        return out
    

class UNet_Resblocks(pl.LightningModule):
    def __init__(self, use_vae=False, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256],learning_rate=5e-4,pos_weight=None, lambda_=0.0, dropout=0.0, weight_decay=1e-5): # features does not include bottleneck, assumes each index is double of previous
        super().__init__()
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['vae'])
        
        #####VAE part
        self.use_vae = use_vae
        if self.use_vae:    
            self.vae =vae
            self.vae.eval() # set VAE to eval mode
            for param in self.vae.parameters():
                param.requires_grad = False  # freeze VAE parameters
        #####VAE part end
        self.encoder= nn.ModuleList()
        self.decoder= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.learning_rate= learning_rate
        self.weight_decay = weight_decay
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lambda_ = lambda_
        self.dropout = dropout

        # Encoder
        for feature in features:
            self.encoder.append(ResBlock(in_channels, feature, dropout=dropout))
            in_channels= feature

        self.bottleneck= ResBlock(features[-1], features[-1]*2, dropout=dropout)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResBlock(feature*2, feature, dropout=dropout))
        
        self.final_conv= nn.Conv2d(features[0], out_channels, kernel_size=1)
        
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
        optimizer= torch.optim.Adam(self.parameters(), lr= self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def encode_source(self,x):
        # x: input image tensor
        # Use batches for encoding
        if self.use_vae:
            batch_size = x.size(0)
            sub_batches = 64  # number of sub-batches to split into
            latents = []
            with torch.no_grad():
                for i in range(0, batch_size, sub_batches):
                    sub_batch = x[i:i+sub_batches]
                    latent = encode_img(sub_batch)  # get the mean latent vector
                    latents.append(latent)
            latent = torch.cat(latents, dim=0)
            return latent
        else:
            return x
        
    def training_step(self, train_batch, batch_idx):
        x= train_batch['source']
        y=train_batch['target']

        # Encode x with VAE on GPU
        x_encoded = self.encode_source(x)

        y_hat= self(x_encoded)
        bce = self.bce_loss(y_hat, y)
        dice = dice_loss(y_hat, y)
        loss = bce + self.lambda_ * dice
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_bce', bce, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x= val_batch['source']
        y= val_batch['target']

        # Encode x with VAE on GPU
        x_encoded = self.encode_source(x)
        y_hat= self(x_encoded)
        bce= self.bce_loss(y_hat, y)
        dice = dice_loss(y_hat, y)
        loss = bce + self.lambda_ * dice
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_bce', bce, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)

        ######### For visualization, log images 
        # Log images from different batches to avoid showing only augmentations of same base image
        # With ~100 augmentations per image and batch_size=128, each batch contains mostly same base images
        # So we sample from batches 0, 10, 20, 30 to get diversity
        if batch_idx in [0, 10, 20, 30]:
            pred = torch.sigmoid(y_hat)  # Apply sigmoid to get probabilities
        
            # Log 1 image from each batch to get 4 diverse images total
            if x.shape[0] > 0:
                i = 0  # Just take first image from each of the 4 batches
                img_idx = batch_idx // 10  # 0, 1, 2, 3
                
                self.logger[0].experiment.add_image(
                    f'val/image_{img_idx}_source', 
                    x[i], 
                    self.current_epoch
                )

                if self.use_vae:
                    latent_vis = x_encoded[i].mean(dim=0, keepdim=True)  # Average 4 channels to 1
                    self.logger[0].experiment.add_image(f'val/image_{img_idx}_latent', latent_vis, self.current_epoch)

                self.logger[0].experiment.add_image(
                    f'val/image_{img_idx}_target', 
                    y[i], 
                    self.current_epoch
                )
                self.logger[0].experiment.add_image(
                    f'val/image_{img_idx}_prediction', 
                    pred[i], 
                    self.current_epoch
                )
        ##############
    
        return loss
    



    


