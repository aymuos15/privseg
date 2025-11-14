import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from dataset import HAM10000Dataset


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

    # Randomly sample with different seed than training
    random.seed(seed)
    sampled = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
    return sampled


def inference():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    image_dirs = [
        os.path.join(project_root, 'data/HAM10000_images_part_1'),
        os.path.join(project_root, 'data/HAM10000_images_part_2')
    ]
    mask_dir = os.path.join(project_root, 'data/HAM10000_segmentations_lesion_tschandl')
    model_path = os.path.join(project_root, 'checkpoints/unet_model.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first")
        return

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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Running inference on {len(combined_dataset)} images...")
    print(f"Using device: {device}")

    # Inference
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            predictions = (outputs > 0.5).float()

            # Calculate metrics
            dice = dice_coefficient(predictions, masks)
            iou = iou_score(predictions, masks)

            dice_scores.append(dice)
            iou_scores.append(iou)

    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Number of test images: {len(combined_dataset)}")
    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print("="*50)


if __name__ == '__main__':
    inference()
