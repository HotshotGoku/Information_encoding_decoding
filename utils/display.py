
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from utils.config import FPATH


"""Function to convert tensor to PIL images"""
# def tensor_to_pil_v2(tensor):
#     tensor = tensor.permute(1,2,0)  # Convert to  (height, width, channels)
#     return [Image.fromarray((img.numpy() * 255).astype('uint8')) for img in tensor]

def tensor_to_pil_v2(tensor):
    tensor = tensor.permute(1, 2, 0)  # Convert to (height, width, channels)
    img = (tensor.cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(img.squeeze())

def display_predicted_images(input_seed,final_patterns,pred_images, num_samples, order=[0,1,2]):
    """ Function to display 3 images in a grid"""

   
    fig, ax = plt.subplots( 3,num_samples, figsize=(6*num_samples/3,6), layout='constrained')  


    # plt.subplots_adjust(wspace=0.001, hspace=0.001)

    for i in range(num_samples):

     
        image_i=tensor_to_pil_v2(input_seed[i,:,:,:])
        image_o=tensor_to_pil_v2(final_patterns[i,:,:,:])
        image_p= tensor_to_pil_v2(pred_images[i,:,:,:].to("cpu"))
      
        image_list=[image_i,image_o,image_p]
        

        ax[0,i].imshow(image_list[order[0]],cmap='gray')
        ax[0,i].axis('off')

        ax[1,i].imshow(image_list[order[1]],cmap='gray')
        ax[1,i].axis('off')

        ax[2,i].imshow(image_list[order[2]],cmap="gray")
        ax[2,i].axis('off')
      
        print(np.array(image_o).shape)
    
    
    plt.show()



def display_predicted_images_5X(input_seed,actual_latents,pred_latents,pred_images,final_patterns, num_samples):
   
    fig, ax = plt.subplots(5,num_samples, figsize=(6.8, 8))  
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.001, hspace=0.001)


    for i in range(num_samples):

        image_i=tensor_to_pil_v2(input_seed[i,:,:,:])
        image_o=tensor_to_pil_v2(final_patterns[i,:,:,:])
        image_p= tensor_to_pil_v2(pred_images[i,:,:,:].to("cpu"))

        image_al=tensor_to_pil_v2(actual_latents[i,:,:,:].to("cpu"))
        image_pl=tensor_to_pil_v2(pred_latents[i,:,:,:].to("cpu"))
       
        
        ax[0,i].imshow(image_i,cmap='gray')
        ax[0,i].get_xaxis().set_visible(False)
        ax[0,i].get_yaxis().set_visible(False)
      

        ax[1,i].imshow(image_al,cmap='gray')
        
        ax[1,i].get_xaxis().set_visible(False)
        ax[1,i].get_yaxis().set_visible(False)
   

        ax[2,i].imshow(image_pl,cmap="gray")
        
        ax[2,i].get_xaxis().set_visible(False)
        ax[2,i].get_yaxis().set_visible(False)
       
        ax[3,i].imshow(image_p,cmap="gray")
        
        ax[3,i].get_xaxis().set_visible(False)
        ax[3,i].get_yaxis().set_visible(False)
        # ax.set_title('Absolute difference')


        
        ax[4,i].imshow(image_o,cmap="gray")
        
        ax[4,i].get_xaxis().set_visible(False)
        ax[4,i].get_yaxis().set_visible(False)

        print(np.array(image_p).shape)
    
      
    plt.show()


'''
Functions for computing and displaying SSIM comparisions from predicted and actual images

'''

def compute_ssim_high_precision(original_images, reconstructed_images):
    ssim_values = []
    batch_size = original_images.shape[0]

    for i in range(batch_size):
        # Convert tensors to numpy arrays
        original = original_images[i].cpu().numpy()  # Shape: [channels, height, width]
        reconstructed = reconstructed_images[i].cpu().numpy()

        # Ensure images are in the range [0, 1]
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)

        # Convert to [height, width, channels]
        original = np.transpose(original, (1, 2, 0))  # Shape: [height, width, channels]
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

        # Convert reconstructed RGB image to grayscale using standard luminance formula
        if reconstructed.shape[2] == 3:
            reconstructed_gray = np.dot(reconstructed[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            reconstructed_gray = reconstructed.squeeze(axis=2)

        # Ensure original image is grayscale
        if original.shape[2] == 1:
            original_gray = original.squeeze(axis=2)
        else:
            original_gray = np.dot(original[..., :3], [0.2989, 0.5870, 0.1140])

        # Compute SSIM
        ssim_index = ssim(original_gray, reconstructed_gray, data_range=1.0)
        ssim_values.append(ssim_index)

    return ssim_values


def display_images_with_ssim(reference_images, comparison_images,num_samples=3):
    # Compute SSIM values using high-precision data
    ssim_values = compute_ssim_high_precision(reference_images, comparison_images)

    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 4, 16))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    for i in range(num_samples):
        # Get the reference and comparison images
        ref_image = reference_images[i].cpu().numpy()  # Shape: [C, H, W] or [H, W]
        comp_image = comparison_images[i].cpu().numpy()  # Shape: [C, H, W]

        print(f"Sample {i}: ref_image.shape = {ref_image.shape}, comp_image.shape = {comp_image.shape}")

        # Handle the reference image (ground truth)
        if ref_image.ndim == 3 and ref_image.shape[0] == 1:
            # Grayscale image, squeeze the channel dimension
            ref_image_disp = ref_image.squeeze(0)  # Shape: [H, W]
        elif ref_image.ndim == 3:
            # If the image has more channels, convert to grayscale
            ref_image_disp = np.transpose(ref_image, (1, 2, 0))
            ref_image_disp = np.dot(ref_image_disp[..., :3], [0.2989, 0.5870, 0.1140])
        elif ref_image.ndim == 2:
            # Already a 2D grayscale image
            ref_image_disp = ref_image  # Shape: [H, W]
        else:
            raise ValueError(f"Unexpected ref_image shape: {ref_image.shape}")

        # Handle the comparison image (predicted image)
        if comp_image.ndim == 3 and comp_image.shape[0] == 3:
            # Transpose to [H, W, C]
            comp_image_disp = np.transpose(comp_image, (1, 2, 0))
            # Convert RGB to grayscale
            comp_image_disp = np.dot(comp_image_disp[..., :3], [0.2989, 0.5870, 0.1140])
        elif comp_image.ndim == 3 and comp_image.shape[0] == 1:
            comp_image_disp = comp_image.squeeze(0)  # Shape: [H, W]
        elif comp_image.ndim == 2:
            comp_image_disp = comp_image  # Shape: [H, W]
        else:
            raise ValueError(f"Unexpected comp_image shape: {comp_image.shape}")

        # Triplicate the single-channel array along the third dimension
        
        image_o=tensor_to_pil_v2(reference_images[i,:,:,:])
        image_p= tensor_to_pil_v2(comparison_images[i,:,:,:].to("cpu"))


        # Triplicate the single-channel array along the third dimension
        image_o_3channel = np.stack([image_o] * 3, axis=-1)
      
        abs_error = np.abs(np.array(image_o_3channel, dtype=np.float32) - np.array(image_p, dtype=np.float32))
        abs_error_image = Image.fromarray(abs_error.astype(np.uint8))
        
        

        # Display the reference image
        axes[0, i].imshow(ref_image_disp, cmap='gray')
        axes[0, i].axis('off')
        # axes[0, i].set_title('Ground Truth')

        # Display the comparison image
        axes[1, i].imshow(comp_image_disp, cmap='gray')
        axes[1, i].axis('off')
        # axes[1, i].set_title('Predicted Image')


        # Display the absolute error image
        axes[2, i].imshow(abs_error_image, cmap='gray')
        axes[2, i].axis('off')
        # axes[1, i].set_title('Predicted Image')


        # Display the SSIM score
        axes[3, i].text(0.5, 0.5, f'{ssim_values[i]:.3f}',
                        ha='center', va='center', fontsize=50,font=FPATH)
        axes[3, i].axis('off')

    plt.tight_layout()
    plt.show()



def display_images_with_ssim_3rows(reference_images, comparison_images, num_samples=3):
    fig, axes = plt.subplots(3,num_samples, figsize=(num_samples * 4,12),layout='constrained')
    # fig.subplots_adjust(hspace=0.3, wspace=0.5)

    for i in range(num_samples):
        ref_image = tensor_to_pil_v2(reference_images[i])
        comp_image = tensor_to_pil_v2(comparison_images[i])

        # print(type(ref_image))
        # print(type(comp_image))
        
        # Compute SSIM score
        ref_array = np.array(ref_image.convert('L')) / 255.0  # Convert to grayscale for SSIM
        comp_array = np.array(comp_image.convert('L')) / 255.0
        ssim_score = ssim(ref_array, comp_array, data_range=1.0)  # Specify data_range=1.0 for normalized images

        # Display images and SSIM score
        axes[0, i].imshow(ref_image, cmap='gray')
        axes[0, i].axis('off')
        # axes[i, 0].set_title('Reference')

        axes[1, i].imshow(comp_image, cmap='gray')
        axes[1, i].axis('off')
        # axes[i, 1].set_title('Comparison')

        axes[2, i].text(0.5, 0.5, f' {ssim_score:.3f}', ha='center', va='center', fontsize=60,font=FPATH)
        axes[2, i].axis('off')
        # axes[i, 2].set_title('SSIM Score')
    # plt.tight_layout()
    plt.show()


def display_predictions_multiple_samples(input_images, ground_truths, pred_images_list, num_samples_list):
    num_models = len(pred_images_list)
    num_samples = len(input_images)
    total_images = num_models + 2  # Input image + predictions + ground truth

    # Create subplots: num_samples rows, total_images columns
    fig, axes = plt.subplots(num_samples, total_images, figsize=(total_images * 3, num_samples * 3))

    for s in range(num_samples):
        # Display the input image in the first subplot of the row
        if num_samples == 1:
            ax = axes[0]
        else:
            ax = axes[s, 0]
        image_i = tensor_to_pil_v2(input_images[s])
        ax.imshow(image_i,cmap='gray')
        ax.axis('off')
       
        # if s == 0:
        #     ax.set_title('Pattern 1\n(Input)', fontsize=25)

        # Display the predicted images in the middle subplots
        for i in range(num_models):
            if num_samples == 1:
                ax = axes[i + 1]
            else:
                ax = axes[s, i + 1]
            image_p = tensor_to_pil_v2(pred_images_list[i][s])

            image_p = np.array(image_p)
            image_p = np.mean(image_p[..., :3], axis=-1) 



            ax.imshow(image_p,cmap='gray',vmin=0, vmax=255)
            ax.axis('off')
            # if s == 0:
            #     ax.set_title(f"{num_samples_list[i]}", fontsize=25)

        # Display the ground truth image in the last subplot of the row
        if num_samples == 1:
            ax = axes[-1]
        else:
            ax = axes[s, -1]
        image_gt = tensor_to_pil_v2(ground_truths[s])
        ax.imshow(image_gt,cmap='gray')
        ax.axis('off')
        # if s == 0:
        #     # ax.set_title('Pattern 2\n(Ground truth)', fontsize=25)

    plt.tight_layout()
    plt.show()


def display_images_grid(images, grid_size=(10, 10), figsize=(20, 20)):
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, layout='constrained')
    for idx, ax in enumerate(axes.flatten()):
        # ax.axis('off')
        if idx < len(images):
            img = images[idx]                         # np.ndarray or torch.Tensor
            if isinstance(img, np.ndarray):
                ax.imshow(img,cmap='gray')  
                # h, w = img.shape[:2]
                # ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='black', linewidth=1, zorder=10))
                # ax.set_xticks([]); ax.set_yticks([])                 # no ticks
                # ax.tick_params(left=False, bottom=False)             # no tick marks
                # for s in ax.spines.values(): s.set_visible(False)
                h, w = img.shape[:2]
                ax.set_xlim(-0.5, w-0.5); ax.set_ylim(h-0.5, -0.5)
                ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(left=False, bottom=False)

                # draw 4 crisp borders that won't get clipped
                lines = []
                lines += ax.plot([-0.5, w-0.5], [-0.5, -0.5], 'k-', lw=1, clip_on=False)      # top
                lines += ax.plot([-0.5, w-0.5], [h-0.5, h-0.5], 'k-', lw=1, clip_on=False)    # bottom
                lines += ax.plot([-0.5, -0.5], [-0.5, h-0.5], 'k-', lw=1, clip_on=False)      # left
                lines += ax.plot([w-0.5, w-0.5], [-0.5, h-0.5], 'k-', lw=1, clip_on=False)    # right
                for ln in lines: ln.set_antialiased(False)
            else:                                     # torch.Tensor C×H×W
                
                ax.imshow(tensor_to_pil_v2(img), cmap='gray')
                _, h, w = img.shape
                ax.set_xlim(-0.5, w-0.5); ax.set_ylim(h-0.5, -0.5)
                ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(left=False, bottom=False)

                # 4 borders (no clipping, no AA)
                for x0,y0,x1,y1 in [(-0.5,-0.5, w-0.5,-0.5),    # top
                                    (-0.5,h-0.5, w-0.5,h-0.5),  # bottom
                                    (-0.5,-0.5,-0.5,h-0.5),    # left
                                    (w-0.5,-0.5, w-0.5,h-0.5)]: # right
                    ln, = ax.plot([x0,x1],[y0,y1],'k-',lw=1,clip_on=False)
                    ln.set_antialiased(False)


    # plt.tight_layout(pad=1.0)
    plt.show()
