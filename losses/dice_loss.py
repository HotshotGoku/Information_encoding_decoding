import torch 
def dice_loss(inputs, targets):
    """
    Compute the Dice loss (that is backprop-able) between predicted logits and target masks for binary segmentation.
    Args:
    inputs: Predicted logits from the model (before sigmoid), shape (B, 1, H, W)
    targets: Ground truth binary masks, shape (B, 1, H, W), assumes values are floats 0.0 or 1.0
    Returns:    
    Dice loss averaged over the batch
    """
    smooth = 1.0  # to avoid division by zero

    # Apply sigmoid to logits to get probabilities
    probs = torch.sigmoid(inputs)

    # Compute per sample Dice loss, then average over batch
    batch_size= inputs.size(0)
    loss=0.0
    for i in range(batch_size):
        probs_i= probs[i].view(-1)
        targets_i= targets[i].view(-1)

        # Compute intersection and unions
        intersection = (probs_i * targets_i).sum()
        total = probs_i.sum() + targets_i.sum()

        # Compute Dice coefficient
        dice_coeff = (2.0 * intersection + smooth) / (total + smooth)

        # Dice loss is 1 - Dice coefficient
        loss = loss + (1.0 - dice_coeff)

    avg_loss= loss/batch_size
    return avg_loss

  