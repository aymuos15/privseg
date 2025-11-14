import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from model_mask2former import Mask2FormerBinarySegmentation
from dataset import HAM10000Dataset


def count_parameters(model):
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1.0):
    """Calculate IoU (Intersection over Union)"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def get_image_ids(image_dirs, mask_dir, num_samples=100, seed=123):
    """Get random image IDs that have corresponding masks"""
    all_image_ids = []

    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.endswith('.jpg'):
                    img_id = img_file.replace('.jpg', '')
                    mask_path = os.path.join(mask_dir, f"{img_id}_segmentation.png")
                    if os.path.exists(mask_path):
                        all_image_ids.append((img_id, img_dir))

    # Randomly sample
    random.seed(seed)
    sampled = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
    return sampled


def evaluate_model(model, test_loader, device, model_name="Model"):
    """Evaluate a model on test set"""
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward - handle different model types
            if isinstance(model, UNet):
                outputs = model(images)
                predictions = (outputs > 0.5).float()
            else:  # Mask2Former
                outputs = model.predict(images, threshold=0.5)
                predictions = (outputs > 0.5).float()

            # Calculate metrics
            dice = dice_coefficient(predictions, masks)
            iou = iou_score(predictions, masks)

            dice_scores.append(dice)
            iou_scores.append(iou)

    return {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores)
    }


def compare():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    image_dirs = [
        os.path.join(project_root, 'data/HAM10000_images_part_1'),
        os.path.join(project_root, 'data/HAM10000_images_part_2')
    ]
    mask_dir = os.path.join(project_root, 'data/HAM10000_segmentations_lesion_tschandl')
    unet_path = os.path.join(project_root, 'checkpoints/unet_model.pth')
    mask2former_path = os.path.join(project_root, 'checkpoints/mask2former_model.pth')

    # Check if models exist
    if not os.path.exists(unet_path):
        print(f"Error: U-Net model not found at {unet_path}")
        print("Please run train.py first")
        return

    if not os.path.exists(mask2former_path):
        print(f"Error: Mask2Former model not found at {mask2former_path}")
        print("Please run train_mask2former.py first")
        return

    # Get test data (same seed as inference scripts)
    print("Loading test data...")
    test_data = get_image_ids(image_dirs, mask_dir, num_samples=100, seed=123)

    # Create datasets
    test_dataset_part1 = HAM10000Dataset(
        image_dir=image_dirs[0],
        mask_dir=mask_dir,
        image_ids=[img_id for img_id, img_dir in test_data if img_dir == image_dirs[0]]
    )

    test_dataset_part2 = HAM10000Dataset(
        image_dir=image_dirs[1],
        mask_dir=mask_dir,
        image_ids=[img_id for img_id, img_dir in test_data if img_dir == image_dirs[1]]
    )

    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([test_dataset_part1, test_dataset_part2])
    test_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load U-Net
    print("\nLoading U-Net model...")
    unet = UNet(in_channels=3, out_channels=1).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet_params = count_parameters(unet)

    # Load Mask2Former
    print("Loading Mask2Former model...")
    mask2former = Mask2FormerBinarySegmentation().to(device)
    mask2former.load_state_dict(torch.load(mask2former_path, map_location=device))
    mask2former_params = count_parameters(mask2former)

    # Evaluate both models
    print(f"\nEvaluating on {len(combined_dataset)} test images...")
    print(f"Using device: {device}\n")

    unet_results = evaluate_model(unet, test_loader, device, "U-Net")
    mask2former_results = evaluate_model(mask2former, test_loader, device, "Mask2Former")

    # Print comparison
    print("\n" + "="*70)
    print(" "*20 + "MODEL COMPARISON")
    print("="*70)
    print(f"{'Metric':<30} {'U-Net':<20} {'Mask2Former':<20}")
    print("-"*70)

    unet_param_str = f"{unet_params/1e6:.2f}M"
    m2f_param_str = f"{mask2former_params/1e6:.2f}M"
    print(f"{'Model Parameters':<30} {unet_param_str:<20} {m2f_param_str:<20}")
    print("-"*70)

    unet_dice_str = f"{unet_results['dice_mean']:.4f} ± {unet_results['dice_std']:.4f}"
    m2f_dice_str = f"{mask2former_results['dice_mean']:.4f} ± {mask2former_results['dice_std']:.4f}"
    print(f"{'Dice Score (mean ± std)':<30} {unet_dice_str:<20} {m2f_dice_str:<20}")

    unet_iou_str = f"{unet_results['iou_mean']:.4f} ± {unet_results['iou_std']:.4f}"
    m2f_iou_str = f"{mask2former_results['iou_mean']:.4f} ± {mask2former_results['iou_std']:.4f}"
    print(f"{'IoU Score (mean ± std)':<30} {unet_iou_str:<20} {m2f_iou_str:<20}")

    print("="*70)

    # Print winner
    print("\nPerformance Summary:")
    dice_winner = "U-Net" if unet_results['dice_mean'] > mask2former_results['dice_mean'] else "Mask2Former"
    iou_winner = "U-Net" if unet_results['iou_mean'] > mask2former_results['iou_mean'] else "Mask2Former"
    param_winner = "U-Net" if unet_params < mask2former_params else "Mask2Former"

    print(f"  - Best Dice Score: {dice_winner}")
    print(f"  - Best IoU Score: {iou_winner}")
    print(f"  - Fewer Parameters: {param_winner}")
    print()


if __name__ == '__main__':
    compare()
