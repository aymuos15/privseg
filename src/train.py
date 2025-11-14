import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from dataset import HAM10000Dataset


def get_image_ids(image_dirs, mask_dir, num_samples=100):
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
    random.seed(42)
    sampled = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
    return sampled


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()


def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient for evaluation"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def train():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    image_dirs = [
        os.path.join(project_root, 'data/HAM10000_images_part_1'),
        os.path.join(project_root, 'data/HAM10000_images_part_2')
    ]
    mask_dir = os.path.join(project_root, 'data/HAM10000_segmentations_lesion_tschandl')

    print("Loading training data...")
    train_data = get_image_ids(image_dirs, mask_dir, num_samples=100)

    part1_ids = [img_id for img_id, img_dir in train_data if img_dir == image_dirs[0]]
    part2_ids = [img_id for img_id, img_dir in train_data if img_dir == image_dirs[1]]

    train_dataset = HAM10000Dataset(
        image_dir=image_dirs[0],
        mask_dir=mask_dir,
        image_ids=part1_ids
    )

    train_dataset_part2 = HAM10000Dataset(
        image_dir=image_dirs[1],
        mask_dir=mask_dir,
        image_ids=part2_ids
    )

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_part2])
    train_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce_loss = nn.BCELoss()

    # Training
    num_epochs = 2
    print(f"Training on {len(combined_dataset)} images for {num_epochs} epochs...")
    print(f"Using device: {device}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice_scores = []

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate Dice score
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                for i in range(predictions.shape[0]):
                    dice = dice_coefficient(predictions[i], masks[i])
                    epoch_dice_scores.append(dice)

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = sum(epoch_dice_scores) / len(epoch_dice_scores)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")

    # Save model
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'unet_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == '__main__':
    train()
