# Peformance evaluation of different models
# Currently the model predicts the seed images, and seeds can be in number from about 1-20 seeds, some OOD cases in test set might have more
# We will evaluation the performance on the entire test set (about 132 images)

# Metrics: 
# 1. MSE loss between predicted seed image and ground truth seed image (pixel-wise comparison)  
# 2. DICE loss between predicted seed image and ground truth seed image (overlap of predicted and true seed regions)
# 3. Tolerance Precision, Recall and F1 score, Measuring pixel wise true and false positives, and true and false negatives*
# 4. Pixel accuracy, measuring the percentage of correctly predicted pixels (both seed and non-seed)

# * Point matching with tolerance radius r, can use either Hungarian matching or greedy matching, to match predicted seed points to true seed points

import torch
import matplotlib.pyplot as plt
import torch 
from scipy.optimize import linear_sum_assignment
import numpy as np 


# MSE LOSS
def mse_loss(predicted_seed_image, true_seed_image):
    # Ensure the input images are of the same shape
    assert predicted_seed_image.shape == true_seed_image.shape, "Input images must have the same shape"
    
    # Convert to float and normalize to [0, 1] to match training
    predicted_seed_image = predicted_seed_image.float() / 255.0
    true_seed_image = true_seed_image.float() / 255.0
    
    # Calculate the MSE loss
    loss = torch.mean((predicted_seed_image - true_seed_image) ** 2)
    
    return loss

# DICE loss 
# Modifying the DICE loss function used earlier to work with the predicted and true seed images,which are images with pixel value 255 for seed and 0 for non-seed. 

def dice_loss (predicted_seed_image, true_seed_image, smooth=1e-6):
    # Ensure the input images are of the same shape
    assert predicted_seed_image.shape == true_seed_image.shape, "Input images must have the same shape"
    
    # Convert to float and normalize to [0, 1] to match training
    predicted_seed_image = predicted_seed_image.float() / 255.0
    true_seed_image = true_seed_image.float() / 255.0
    
    # Flatten the images to 1D tensors
    predicted_flat = predicted_seed_image.view(-1)
    true_flat = true_seed_image.view(-1)
    
    # Compute intersection and unions
    intersection = (predicted_flat * true_flat).sum()
    total = predicted_flat.sum() + true_flat.sum()

    # Compute Dice coefficient
    dice_coeff = (2.0 * intersection + smooth) / (total + smooth)

    # Dice loss is 1 - Dice coefficient
    loss = 1.0 - dice_coeff
    
    return loss

# Precision, Recall and F1 score
def precision_recall_f1(predicted_seed_image, true_seed_image):
    # Ensure the input images are of the same shape
    assert predicted_seed_image.shape == true_seed_image.shape, "Input images must have the same shape"
    
    # Flatten the images to 1D tensors
    predicted_flat = predicted_seed_image.view(-1)
    true_flat = true_seed_image.view(-1)
    
    # Calculate True Positives, False Positives, and False Negatives
    TP = ((predicted_flat == 255) & (true_flat == 255)).sum().item()  # Both predicted and true are seed
    FP = ((predicted_flat == 255) & (true_flat == 0)).sum().item()    # Predicted seed but true is non-seed
    FN = ((predicted_flat == 0) & (true_flat == 255)).sum().item()    # Predicted non-seed but true is seed
    
    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP + 1e-6)  # Add small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-6)     # Add small epsilon to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # Add small epsilon to avoid division by zero
    
    return precision, recall, f1_score

# Pixel accuracy 

def pixel_accuracy(predicted_seed_image, true_seed_image):
    # Ensure the input images are of the same shape
    assert predicted_seed_image.shape == true_seed_image.shape, "Input images must have the same shape"
    
    # Flatten the images to 1D tensors
    predicted_flat = predicted_seed_image.view(-1)
    true_flat = true_seed_image.view(-1)
    
    # Calculate the number of correctly predicted pixels
    correct_pixels = (predicted_flat == true_flat).sum().item()
    
    # Calculate total number of pixels
    total_pixels = predicted_flat.numel()
    
    # Calculate pixel accuracy
    accuracy = correct_pixels / total_pixels
    
    return accuracy

# Point matching with tolerance radius r, using Hungarian/greedy matching
 

def point_matching(predicted_seed_image, true_seed_image, tolerance_radius, algorithm='greedy'):
    # Ensure the input images are of the same shape
    assert predicted_seed_image.shape == true_seed_image.shape, "Input images must have the same shape"
    
    # Get the coordinates of predicted and true seed points
    predicted_points = torch.nonzero(predicted_seed_image == 255).cpu().numpy()  # Get coordinates of predicted seed points
    true_points = torch.nonzero(true_seed_image == 255).cpu().numpy()            # Get coordinates of true seed points
    
    if len(predicted_points) == 0 or len(true_points) == 0:
        return 0, 0, 0  # No points to match, return zero precision, recall, and F1 score
    
    # Calculate distance matrix between predicted and true points
    distance_matrix = np.linalg.norm(predicted_points[:, None] - true_points[None, :], axis=-1)
    
    if algorithm == 'greedy':
        # Greedy matching
        matches = []
        for i in range(len(predicted_points)):
            min_distance_index = np.argmin(distance_matrix[i])
            if distance_matrix[i][min_distance_index] <= tolerance_radius:
                matches.append((i, min_distance_index))
                distance_matrix[:, min_distance_index] = np.inf  # Mark this true point as matched
        TP = len(matches)
        FP = len(predicted_points) - TP
        FN = len(true_points) - TP
        
    elif algorithm == 'hungarian':
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matches = [(row_ind[i], col_ind[i]) for i in range(len(row_ind)) if distance_matrix[row_ind[i], col_ind[i]] <= tolerance_radius]
        TP = len(matches)
        FP = len(predicted_points) - TP
        FN = len(true_points) - TP
        
    else:
        raise ValueError("Invalid algorithm specified. Use 'greedy' or 'hungarian'.")
    
    precision = TP / (TP + FP + 1e-6)  # Add small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-6)     # Add small epsilon to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # Add small epsilon to avoid division by zero
    
    return precision, recall, f1_score

# Visualize point matching


def visualize_point_matching(predicted_seed_image, true_seed_image, tolerance_radius, algorithm='greedy', title=''):
    """
    Visualize the point matching between predicted and true seed points.
    Shows matched points connected by lines, unmatched points highlighted.
    """
    # Get coordinates
    predicted_points = torch.nonzero(predicted_seed_image == 255).cpu().numpy()
    true_points = torch.nonzero(true_seed_image == 255).cpu().numpy()
    
    # Calculate distance matrix
    if len(predicted_points) == 0 or len(true_points) == 0:
        print(f"{title}: No points to visualize")
        return
    
    distance_matrix = np.linalg.norm(predicted_points[:, None] - true_points[None, :], axis=-1)
    
    # Perform matching
    if algorithm == 'greedy':
        matches = []
        dist_copy = distance_matrix.copy()
        for i in range(len(predicted_points)):
            min_distance_index = np.argmin(dist_copy[i])
            if dist_copy[i][min_distance_index] <= tolerance_radius:
                matches.append((i, min_distance_index))
                dist_copy[:, min_distance_index] = np.inf
    else:  # hungarian
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matches = [(row_ind[i], col_ind[i]) for i in range(len(row_ind)) 
                   if distance_matrix[row_ind[i], col_ind[i]] <= tolerance_radius]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(predicted_seed_image.cpu().numpy(), cmap='gray', alpha=0.3)
    ax.set_title(f'{title}\n{algorithm.capitalize()} Matching (radius={tolerance_radius}px)')
    
    # Plot matched points and connections
    matched_pred = set([m[0] for m in matches])
    matched_true = set([m[1] for m in matches])
    
    # Draw connections for matched points
    for pred_idx, true_idx in matches:
        py, px = predicted_points[pred_idx]
        ty, tx = true_points[true_idx]
        ax.plot([px, tx], [py, ty], 'g-', linewidth=1.5, alpha=0.6)
    
    # Plot true points (ground truth)
    for idx, (y, x) in enumerate(true_points):
        if idx in matched_true:
            ax.plot(x, y, 'go', markersize=12, markerfacecolor='lime', markeredgecolor='darkgreen', 
                   markeredgewidth=2, label='True (matched)' if idx == list(matched_true)[0] else '')
        else:
            ax.plot(x, y, 'ro', markersize=12, markerfacecolor='red', markeredgecolor='darkred', 
                   markeredgewidth=2, label='True (missed)' if idx == [i for i in range(len(true_points)) if i not in matched_true][0] else '')
    
    # Plot predicted points
    for idx, (y, x) in enumerate(predicted_points):
        if idx in matched_pred:
            ax.plot(x, y, 'bs', markersize=10, markerfacecolor='cyan', markeredgecolor='darkblue', 
                   markeredgewidth=2, label='Pred (matched)' if idx == list(matched_pred)[0] else '')
        else:
            ax.plot(x, y, 'ms', markersize=10, markerfacecolor='magenta', markeredgecolor='purple', 
                   markeredgewidth=2, label='Pred (false alarm)' if idx == [i for i in range(len(predicted_points)) if i not in matched_pred][0] else '')
    
    # Add statistics
    TP = len(matches)
    FP = len(predicted_points) - TP
    FN = len(true_points) - TP
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    stats_text = f'TP={TP}, FP={FP}, FN={FN}\nPrec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    ax.axis('off')
    plt.tight_layout()
    return fig


# Batch visualization for multiple images
def visualize_batch_matching(predicted_images, true_images, tolerance_radius=5, algorithm='greedy', titles=None):
    """
    Visualize point matching for multiple images in a grid.
    
    Args:
        predicted_images: list of predicted seed images (torch tensors)
        true_images: list of true seed images (torch tensors)
        tolerance_radius: matching tolerance in pixels
        algorithm: 'greedy' or 'hungarian'
        titles: optional list of titles for each subplot
    """
    n_images = len(predicted_images)
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n_images)]
    
    for i in range(n_images):
        visualize_point_matching(predicted_images[i], true_images[i], 
                                tolerance_radius, algorithm, titles[i])
        plt.show()