
from diffusers import AutoencoderKL
import torch

'''
Define pre-trained SD VAE for decoding orginal predicted patterns from the latent predicted patterns
dilResnet will predict latent representation of patterns 

'''

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# first load the VAE
# was using CompVis/stable-diffusion-v1-4 earlier,  now switching v1-5 to match with ControlNet pipeline 
vae = AutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae") 

vae.to(device)

def encode_img(input_img):
    # Handle both single images and batches
    original_shape = input_img.shape
    
    # If single image (C, H, W), add batch dimension
    if len(original_shape) == 3:
        if original_shape[0] == 1:  # Grayscale (1, H, W)
            input_img = input_img.repeat(3, 1, 1)  # Convert to RGB
        input_img = input_img.unsqueeze(0)  # Add batch dimension
    elif len(original_shape) == 4:  # Already batched (B, C, H, W)
        if original_shape[1] == 1:  # Grayscale batch
            input_img = input_img.repeat(1, 3, 1, 1)  # Convert to RGB
        # Already has RGB channels or we converted above
    
    # Get the actual device of the VAE (not the hardcoded device variable)
    vae_device = next(vae.parameters()).device
    input_img = input_img.to(vae_device)
    
    with torch.no_grad():
        latent = vae.encode(input_img * 2 - 1)  # Scale to [-1, 1] for VAE
    return 0.18215 * latent.latent_dist.sample()    



def decode_img(latents):
    # bath of latents -> list of images
    # Get the actual device of the VAE (not the hardcoded device variable)
    vae_device = next(vae.parameters()).device
    latents = (1 / 0.18215) * latents.to(vae_device)
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)    # to make outputs from 0 to 1 
    image = image.detach()
    return image
