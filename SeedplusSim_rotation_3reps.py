import os
import numpy as np
import cv2
from utils.preprocess import preprocess_seed
from utils.config import SEED_FOLDER_TEST_INFOENCODING_MORESEEDS, SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS, SIM_FOLDER_TEST_INFOENCODING_MORESEEDS, SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS
import glob

# We will augment the images for both the seed and the simulation folders 
# Each image will be rotated by 2 rotations- 120 degrees and 240 degrees, in addition to the original image (0 degree rotation) - so 3 rotations total per image.
# This is to increase the datasize for training the model by a factor of 3, so that it matches with the experimental pipeline which uses the diffusion model predictions


def process_seed_image(img_path, angle):
    # Coordinate-based rotation method
    
    # Preprocess the seed image using preprocess_seed function # to make 32x32
    # note since we are already preprocessing the image here, we have to make sure no preprocessing at the time of dataset building 
    img = preprocess_seed(img_path, top_crop=0, bottom_crop=0, left_crop=2, right_crop=3, img_length=32, img_width=32)
    
    if img is None:
        print(f"Failed to preprocess image at {img_path}")
        return None
    
    # Binarize
    mask = (img > 0).astype(np.uint8)
    
    # Get seed coordinates (x, y)
    ys, xs = np.where(mask > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (K,2)
    
    # Rotation center (pixel-center for 32x32 image)
    c = np.array([15.5, 15.5], dtype=np.float32)
    
    # Rotation angle
    th = np.deg2rad(float(angle)).astype(np.float32)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=np.float32)
    
    # Apply rotation
    rot = (coords - c) @ R.T + c
    rot = np.rint(rot).astype(int)
    
    # Create output image (grayscale)
    rot_mask = np.zeros((32, 32), dtype=np.uint8)
    ok = (rot[:,0] >= 0) & (rot[:,0] < 32) & (rot[:,1] >= 0) & (rot[:,1] < 32)
    rot_mask[rot[ok,1], rot[ok,0]] = 255
    
    return rot_mask


def process_simulation_image(img, angle):
    """Optimized rotation using OpenCV warpAffine - much faster than pixel-by-pixel method.
    Uses NEGATED angle to match backward mapping direction and INTER_NEAREST interpolation."""
    
    # note here there is no preprocessing involved 
    # so we will proceed with the 

    # Find the center of the image (which is also the center of the inscribed circle)
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    # Radius of the inscribed circle is half the length of the shortest side of the rectangle
    radius = min(center_x, center_y)
    
    # NEGATE the angle because:
    # - Original method uses backward mapping (inverse rotation)
    # - OpenCV uses forward rotation
    # So we need to rotate in the OPPOSITE direction to match!
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    
    # Use INTER_NEAREST to match the original method's int(y), int(x) sampling
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), 
                                  flags=cv2.INTER_NEAREST)
    
    # Create circular mask to keep only pixels within inscribed circle
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply mask
    output_img = cv2.bitwise_and(rotated_img, rotated_img, mask=mask)
    
    return output_img



if __name__ == "__main__":

    augmentation_perimage= 3

    seed_folder = SEED_FOLDER_TEST_INFOENCODING_MORESEEDS
    new_seed_folder= SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS

    simulation_folder= SIM_FOLDER_TEST_INFOENCODING_MORESEEDS
    new_simulation_folder= SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS

    # Ensure the new folders exist
    os.makedirs(new_seed_folder, exist_ok=True)
    os.makedirs(new_simulation_folder, exist_ok=True)

    # select only 30,000 images from the seed and the simulation folders 
    matched_files_total_seed= sorted(glob.glob(os.path.join(seed_folder, "*.png")))[:30000]
    matched_files_total_simulation= sorted(glob.glob(os.path.join(simulation_folder, "*.png")))[:30000]

    # get only the required number of unique samples
    print(f"Number of files to process: {len(matched_files_total_seed)}")
    angle_all = [0, 120, 240]  # 3 rotations: 0°, 120°, 240°

    # Process each pair of images
    for filename_seed, filename_sim in zip(matched_files_total_seed, matched_files_total_simulation):

        # glob returns full paths; extract basenames for output filenames
        basename_seed = os.path.basename(filename_seed)
        basename_sim  = os.path.basename(filename_sim)

        sim_img = cv2.imread(filename_sim)

        for angle in angle_all:

            # Process and rotate the seed image (takes path)
            rotated_seed_img = process_seed_image(filename_seed, angle)
            # Process and rotate the simulation image (takes loaded image array)
            rotated_sim_img = process_simulation_image(sim_img, angle)

            if rotated_seed_img is None:
                continue

            # Save with angle suffix to distinguish the 3 rotations
            new_seed_filename = f"{os.path.splitext(basename_seed)[0]}_rot{int(angle)}{os.path.splitext(basename_seed)[1]}"
            new_sim_filename  = f"{os.path.splitext(basename_sim)[0]}_rot{int(angle)}{os.path.splitext(basename_sim)[1]}"

            cv2.imwrite(os.path.join(new_seed_folder, new_seed_filename), rotated_seed_img)
            cv2.imwrite(os.path.join(new_simulation_folder, new_sim_filename), rotated_sim_img)

    print("Processing completed.")
