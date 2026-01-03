from PIL import Image
import numpy as np
import cv2
import torch

# the following functions preprocess a single image.

def preprocess_simulation(sim_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
    
    # Read image using cv2
    img_array = cv2.imread(sim_path)
    
    if img_array is None:
        print(f"Failed to load image at {sim_path}")
        return None

    #convert to grayscale
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #note: modification on 02 11 2025 to check how it will look with uncropped images

    # top_crop=0
    # bottom_crop=0
    # left_crop=0
    # right_crop=0

    # un modified back , it works, by default using cropping for now
    # for reference see /hpc/dctrl/ks723/inference/v20250211_2231/

    new_height = img_gray.shape[0] - (top_crop + bottom_crop)
    new_width = img_gray.shape[1] - (left_crop + right_crop)

    new_img = img_gray[top_crop:top_crop+new_height, left_crop:left_crop+new_width]
    simulated_image_resized=cv2.resize(new_img,(img_length,img_width))
    return simulated_image_resized


def grayfordisplay(img_path, img_length=256, img_width=256, img_type='sim'):

    """
    Preprocess single image : 
    1. Invert colors, white to black and black to white
    2. Convert black to gray for better visualization
    3. Resize to desired dimensions

    Args:
        img_path: Input image path 
        img_length, img_width: Output dimensions
        use_gentle: Use gentler segmentation for noisy images
    Returns: cv2 image: Processed image of Experimental data.

    """
    
    img=cv2.imread(img_path, cv2.IMREAD_COLOR)
    # convert to RGB
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_inverse = cv2.bitwise_not(np.array(img))
    

    # change artificial color to grey for visualization purposes
    # RGB color code : (70,70,70)
    img_inverse[np.all(img_inverse == [0,0,0], axis=-1)] = [152,152,152]

    if img_type=='seed':
        return img_inverse # for seed images, just invert and change black to gray resize causing weird artifacts

    # remove the small border artifacts by adding a white border
    # Set all channels including alpha to 255 for opaque white borders
    img_inverse[0:2, :] = [255, 255, 255]
    img_inverse[-2:, :] = [255, 255, 255]
    img_inverse[:, 0:2] = [255, 255, 255]
    img_inverse[:, -2:] = [255, 255, 255]

    img_inverse = cv2.resize(img_inverse, (img_length, img_width))  # resize after doing operations

    return img_inverse    




def preprocess_experimental_backgroundwhite(exp_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image(crops, applies white background mask and resizes) from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized background white image as a numpy array converted to RGB.
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_COLOR)  # Read in color to preserve colony colors
    # print(img_array.shape)

   # convert to RGB
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # resize the array
    img_array = cv2.resize(img_array, (img_width, img_length))

    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    ########################
    # do some cropping , note do appropriate cropping for simulation images too
    #########################

    new_height = img_array.shape[0] - (top_crop + bottom_crop)
    new_width = img_array.shape[1] - (left_crop + right_crop)

    new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # now change black to white for the masked area
    new_array_i[np.all(new_array_i == [0,0,0], axis=-1)] = [255,255,255]  # change black to white

    # resize after cropping
    new_array_i = cv2.resize(new_array_i, (img_width, img_length))

    return new_array_i


def preprocess_experimental_backgroundwhite_blackandwhite(exp_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image from the given path (crops, applies white mask and thresholds) and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized black and white image as a numpy array converted to grayscale
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_COLOR)  # Read in color to preserve colony colors
    # print(img_array.shape)

    # convert to RGB
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # resize the array
    img_array = cv2.resize(img_array, (img_width, img_length))

    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    ########################
    # do some cropping , note do appropriate cropping for simulation images too
    #########################

    new_height = img_array.shape[0] - (top_crop + bottom_crop)
    new_width = img_array.shape[1] - (left_crop + right_crop)

    new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # now change black to white for the masked area
    new_array_i[np.all(new_array_i == [0,0,0], axis=-1)] = [255,255,255]  # change black to white

    # resize after cropping
    new_array_i = cv2.resize(new_array_i, (img_width, img_length))

    #Convert to grayscale and apply thresholding
    img_gray_adj = cv2.cvtColor(new_array_i, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray_adj, (5, 5), 0)
    _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

    return img_thresh



def preprocess_simulation_graybackground(sim_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
    
    """
    Preprocess a single image from the given path (crops and applies grayback color to patterns) and return a PIL-compatible output.

    Parameters:
        sim_path (str): Path to the simulation image file, assumes simulation is unprocessed with extra space with no growth 
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for simulation images in accordance with experimental images
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized grayscale image as a numpy array.

    """


    # Read image using cv2
    img_array_i = cv2.imread(sim_path,cv2.IMREAD_GRAYSCALE)

    if img_array_i is None:
        print(f"Failed to load image at {sim_path}")
        return None


    new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
    new_width = img_array_i.shape[1] - (left_crop + right_crop)

    new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

   
    # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
    # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
    new_array_i=cv2.resize(new_array_i,(img_length,img_width))
    # print(new_array_i.shape)

    img_inverse = cv2.bitwise_not(np.array(new_array_i))
    
    # 2. Change artificial color to gray for visualization
    # 2.1 Extra thresholding step for predicted images to remove noise
    img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
    # 2.2 Change black to gray (for grayscale images)
    img_inverse[img_inverse == 0] = 152


    return img_inverse


def preprocess_seed_graybackground(seed_path,top_crop=0, bottom_crop=0, left_crop=0, right_crop=0):
    
    """
    Preprocess simulation input images in grayscale from raw simulation data.

    Args:
        seed_path (str): Input image path 
        top_crop, bottom_crop, left_crop, right_crop (int): Number of pixels to crop from each side

    
    Returns: PIL.Image: Grayscale image of Simulation data. 
    """

    # note: no resizing for input seed images, images are already img_length=32, img_width=32
    
    img_array_i = cv2.imread(seed_path,cv2.IMREAD_GRAYSCALE)
    if img_array_i is None:
        print(f"Failed to load image at {seed_path}")
        return None
    

    new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
    new_width = img_array_i.shape[1] - (left_crop + right_crop)
    new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # OTSU thresholding 
    (T, new_array_i) = cv2.threshold(new_array_i, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

    # Convert images to gray on light background for better visualization
    img_inverse = cv2.bitwise_not(np.array(new_array_i))
    
    #  Change artificial color to gray for visualization
    #  Extra thresholding step for predicted images to remove noise
    img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
    #  Change black to gray (for grayscale images)
    img_inverse[img_inverse == 0] = 152
    return img_inverse

def preprocess_experimental_backgroundblack_blackandwhite(exp_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image from the given path (crops and thresholds) and returns a PIL-compatible output.

    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized black and white image as a numpy array converted to grayscale.
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_COLOR)  # Read in color to preserve colony colors
    # print(img_array.shape)

    # convert to RGB
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # resize the array
    img_array = cv2.resize(img_array, (img_width, img_length))

    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    ########################
    # do some cropping , note do appropriate cropping for simulation images too
    #########################

    new_height = img_array.shape[0] - (top_crop + bottom_crop)
    new_width = img_array.shape[1] - (left_crop + right_crop)

    new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # resize after cropping
    new_array_i = cv2.resize(new_array_i, (img_width, img_length))

    #Convert to grayscale and apply thresholding
    img_gray_adj = cv2.cvtColor(new_array_i, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray_adj, (5, 5), 0)
    _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

    return img_thresh


def preprocess_experimental_backgroundblack_blackandwhite_nocropping(exp_path,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image from the given path (converts white plate background to black) thresholds and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is white
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized black and white image as a numpy array converted to grayscale
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_COLOR)  # Read in color to preserve colony colors
    # print(img_array.shape)

    # convert to RGB
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # resize the array
    img_array = cv2.resize(img_array, (img_width, img_length))

    mask = np.all(img_array >= 230, axis=-1)   # True where R,G,B all ≥ 230
    img_array[mask] = [0, 0, 0]  # Set those pixels to black


    # Convert to grayscale and apply thresholding
    img_gray_adj = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray_adj, (5, 5), 0)
    _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

    return img_thresh


#######################################################################

# These functions are not used in the current paper version, kept for future reference. 

########################################################################

def preprocess_experimental_backgroundwhite_segmentedcolonies(exp_path, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image from the given path (does image segmentation and crops) and return a PIL-compatible output.
    Note: Not used in the current paper version, kept for future reference.
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized background white image as a numpy array converted to RGB.
    """
    # Read image using cv2
    img_array_i = cv2.imread(exp_path,cv2.IMREAD_COLOR)  #

    if img_array_i is None:
        print(f"Failed to load image at {exp_path}")
        return None

   # convert to RGB
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

   ########################
    g   = cv2.cvtColor(img_array_i, cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(g, (5,5), 0)
    _, th= cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)    # or THRESH_BINARY_INV if needed

    alpha = (th/255.0)[..., None]                              # HxWx1 mask
    out_bgr = (alpha * img_array_i + (1-alpha) * 255).astype(np.uint8)  # white bg, keep colony colors
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    #########################
    img_array_i=out_rgb

    # resize the array
    img_array_i = cv2.resize(img_array_i, (img_width, img_length))

    new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
    new_width = img_array_i.shape[1] - (left_crop + right_crop)

    new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # resize after cropping
    new_array_i = cv2.resize(new_array_i, (img_width, img_length))


    return new_array_i


def process_tensor_batch_with_grayfordisplay(tensor_batch, img_length=256, img_width=256, img_type='sim'):
    """
    Process a batch of torch tensors using grayfordisplay logic directly
    
    Args:
        tensor_batch: torch.Tensor of shape (B, C, H, W) where C can be 1 or 3
        img_length, img_width: Output dimensions
        img_type: 'sim' or 'seed'
    
    Returns:
        torch.Tensor: Processed images in same batch format as input (B, C, H, W)
    """
    batch_size, channels = tensor_batch.shape[0], tensor_batch.shape[1]
    processed_images = []
    
    for i in range(batch_size):
        # Extract single image from batch: (C, H, W)
        single_tensor = tensor_batch[i].cpu().numpy()
        
        # Handle both 1-channel and 3-channel inputs
        if channels == 1:
            # Convert grayscale to RGB: (1, H, W) -> (H, W, 3)
            img = single_tensor[0]  # Remove channel dimension
            img = np.stack([img, img, img], axis=-1)  # Convert to 3-channel
        else:
            # Convert RGB: (3, H, W) -> (H, W, 3)
            img = single_tensor.transpose(1, 2, 0)
        
        # Ensure proper data type and range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Apply grayfordisplay logic 
     
        # 1. Invert colors
        img_inverse = cv2.bitwise_not(np.array(img))
        
        # 2. Change artificial color to gray for visualization
        # 2.1 Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        # 2.2 Change black to gray
        img_inverse[np.all(img_inverse == [0,0,0], axis=-1)] = [152,152,152]
        
        if img_type != 'seed':
            # 3. Remove border artifacts by adding white border
            img_inverse[0:2, :] = [255, 255, 255]
            img_inverse[-2:, :] = [255, 255, 255]
            img_inverse[:, 0:2] = [255, 255, 255]
            img_inverse[:, -2:] = [255, 255, 255]
            
            # 4. Resize after operations
            img_inverse = cv2.resize(img_inverse, (img_length, img_width))
        
        # Convert back to tensor and normalize to [0, 1]
        processed_tensor = torch.from_numpy(img_inverse.astype(np.float32) / 255.0)
        
        # Convert back to original format
        if channels == 1:
            # Convert back to grayscale: (H, W, 3) -> (1, H, W)
            processed_tensor = processed_tensor.mean(dim=-1, keepdim=True).permute(2, 0, 1)
        else:
            # Convert back to RGB: (H, W, 3) -> (3, H, W)
            processed_tensor = processed_tensor.permute(2, 0, 1)
        
        processed_images.append(processed_tensor)
    
    # Stack back into batch tensor
    processed_batch = torch.stack(processed_images, dim=0)
    
    # Ensure same device as input
    return processed_batch.to(tensor_batch.device)


def preprocess_experimental_initialstage_mod(img_path, img_length=256, img_width=256):
    """
    Preprocess single image in color from raw experimental data.

    Args:
        img_path: Input image path 
        img_length, img_width: Output dimensions
    

    Returns: PIL.Image: Color and Cropped image of Experimental data. Cropped exterior is a white masked circle. 
    """

    img = cv2.imread(img_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=20, maxRadius=600)

    # Assuming the first detected circle is the plate (adjust accordingly)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            # Define a new radius for the mask that is 20 pixels smaller than the detected radius
            new_radius = radius - 82

            # Create a mask where all values are set to zeros (black)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)   

            # Draw a filled white circle on the mask where the new, smaller circle is
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

            # Apply the mask to the original image (set pixels outside the new circle to white)
            img_masked = cv2.bitwise_and(img, img, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 50    # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)


            # Convert BGR to RGB for plotting
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

            # resize both the adjusted image and the mask
            img_resized = cv2.resize(img_masked_rgb, (img_length, img_width))
            mask_resized = cv2.resize(mask,(img_length, img_width),interpolation=cv2.INTER_NEAREST)

            # re-mask the resized image to force a black border outside the circle
            img_masked_resized = cv2.bitwise_and(img_resized,img_resized,mask=mask_resized)

            # Convert black border to white border (quick fix without changing bitwise operations)
            # Create inverse mask for border areas
            border_mask = mask_resized == 0
            img_masked_resized[border_mask] = [255, 255, 255]  # Set border to white

            return img_masked_resized

    else:
        print("No circles were found")
        return None



def preprocess_inference_thresholded (img_path, img_length=256,img_width=256):
    """
    Preprocess single image in color from inference folder

    Accepts: Input image path, color plate boundary cropped from the inference folder.

    Returns: PIL.Image: Thresholded image of Experimental data

    """

    img = cv2.imread(img_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Apply binary thresholding to the blurred image
    _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

    # Resize the thresholded image
    img_resized = cv2.resize(img_thresh, (img_length, img_width))

    return img_resized

def preprocess_experimental_fromscratch(exp_path, top_crop=30, bottom_crop=30, left_crop=31, right_crop=30,
                                     img_length=256, img_width=256):
    """
    Preprocess image from raw experimental data. 
    
    Parameters:
        exp_path (str): Path to the experimental image file.
        top_crop (int): Pixels to crop from the top.
        bottom_crop (int): Pixels to crop from the bottom.
        left_crop (int): Pixels to crop from the left.
        right_crop (int): Pixels to crop from the right.
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        PIL.Image: BnW and Cropped image of Experimental data.
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path)
    
    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    # Convert to grayscale for edge and circle detection
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            new_radius = radius - 82
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), new_radius, 255, -1)

            # Apply the mask to the original image
            img_masked = cv2.bitwise_and(img_array, img_array, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control
            beta = 50    # Brightness control
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

            # Convert to grayscale and apply thresholding
            img_gray_adj = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img_gray_adj, (5, 5), 0)
            _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

            # Crop and resize the image
            new_height = img_thresh.shape[0] - (top_crop + bottom_crop)
            new_width = img_thresh.shape[1] - (left_crop + right_crop)
            img_cropped = img_thresh[top_crop:top_crop + new_height, left_crop:left_crop + new_width]
            img_resized = cv2.resize(img_cropped, (img_length, img_width))

            # # Convert the final output to PIL format
            # final_image_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB))
            return img_resized
    else:
        print("No circles were found")
        return None


def preprocess_vanillaresize(sim_path, img_length=256, img_width=256):
    
    # Read image using cv2
    img_array = cv2.imread(sim_path)

    if img_array is None:
        return None
    if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    
    seed_image_resized=cv2.resize(img_array,(img_length,img_width))
    return seed_image_resized


def preprocess_experimental(exp_path, top_crop=30, bottom_crop=30, left_crop=31, right_crop=30,
                                     img_length=256, img_width=256):
    """
    Preprocess a single image from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file.
        top_crop (int): Pixels to crop from the top.
        bottom_crop (int): Pixels to crop from the bottom.
        left_crop (int): Pixels to crop from the left.
        right_crop (int): Pixels to crop from the right.
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized grayscale image
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_GRAYSCALE) 
    
    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    ########################
    ## added on 030524
    blur = cv2.GaussianBlur(img_array,(5,5),0)   
    ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
    #########################
    img_array=th3

    new_height = img_array.shape[0] - (top_crop + bottom_crop)
    new_width = img_array.shape[1] - (left_crop + right_crop)

    new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]


    # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
    # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
    experimental_image_resized=cv2.resize(new_array_i,(img_length,img_width))
    return experimental_image_resized

    

def preprocess_experimental_initialstage(img_path, img_length=256,img_width=256):

    """
    Preprocess single image in color from raw experimental data.

    Accepts: Input image path, raw experimental data 

    Returns: PIL.Image: Color and Cropped image of Experimental data. Cropped exterior is a black masked circle. 

    """

    img = cv2.imread(img_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=20, maxRadius=600)

    # print(circles)

    # Assuming the first detected circle is the plate (adjust accordingly)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            # Define a new radius for the mask that is 20 pixels smaller than the detected radius
            new_radius = radius -82
            # print(new_radius)

            # Create a mask where all values are set to zero (black)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # Draw a filled white circle on the mask where the new, smaller circle is
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

            # Apply the mask to the original image (set pixels outside the new circle to black)
            img_masked = cv2.bitwise_and(img, img, mask=mask)


            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 50    # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

            

            # Convert BGR images to RGB for plotting with matplotlib
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
            # output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

            # output_img_grayscale=cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(output_img_grayscale,(5,5),0)   
            # ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
            # ########################
            # img_array_i=th3


            # resize both the adjusted image and the mask
            img_resized = cv2.resize(img_masked_rgb, (img_length, img_width))
            mask_resized = cv2.resize(mask,(img_length, img_width),interpolation=cv2.INTER_NEAREST)

            # re-mask the resized image to force a black border outside the circle
            img_masked_resized = cv2.bitwise_and(img_resized,img_resized,mask=mask_resized)
            return img_masked_resized
    else:
        print("No circles were found")
        return None


def preprocess_seed(seed_path,top_crop=0, bottom_crop=0, left_crop=0, right_crop=0, img_length= 256, img_width=256):
    
    """
    Preprocess simulation input images in grayscale from raw simulation data.

    Args:
        seed_path (str): Input image path 
        top_crop, bottom_crop, left_crop, right_crop (int): Number of pixels to crop from each side

    
    Returns: PIL.Image: Grayscale image of Simulation data. 
    """

    # note: no resizing for input seed images, images are already img_length=32, img_width=32
    
    img_array_i = cv2.imread(seed_path,cv2.IMREAD_GRAYSCALE)
    if img_array_i is None:
        print(f"Failed to load image at {seed_path}")
        return None
    

    new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
    new_width = img_array_i.shape[1] - (left_crop + right_crop)
    new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

    # OTSU thresholding 
    (T, new_array_i) = cv2.threshold(new_array_i, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

    # resize 
    new_array_i = cv2.resize(new_array_i, (img_width, img_length), interpolation=cv2.INTER_NEAREST)

    return new_array_i