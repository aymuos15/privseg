import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_mask2former import Mask2FormerBinarySegmentation
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


def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient for evaluation"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def prepare_mask2former_targets(masks, num_queries=100):
    """
    Prepare target masks and labels for Mask2Former training

    Args:
        masks: Ground truth binary masks [B, 1, H, W]
        num_queries: Number of object queries in Mask2Former

    Returns:
        mask_labels: [B, num_queries, H, W]
        class_labels: [B, num_queries]
    """
    batch_size, _, h, w = masks.shape
    device = masks.device

    mask_labels = []
    class_labels = []

    for i in range(batch_size):
        binary_mask = masks[i, 0]  # [H, W]

        # Create query masks: first query gets the lesion mask, rest are background
        query_masks = torch.zeros((num_queries, h, w), device=device)
        query_classes = torch.zeros(num_queries, dtype=torch.long, device=device)

        # First query: lesion mask
        query_masks[0] = binary_mask
        query_classes[0] = 1  # Lesion class

        # Remaining queries: background (inverse of lesion)
        background_mask = 1.0 - binary_mask
        for q in range(1, num_queries):
            query_masks[q] = background_mask
            query_classes[q] = 0  # Background class

        mask_labels.append(query_masks)
        class_labels.append(query_classes)

    mask_labels = torch.stack(mask_labels)  # [B, num_queries, H, W]
    class_labels = torch.stack(class_labels)  # [B, num_queries]

    return mask_labels, class_labels


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
    train_loader = DataLoader(combined_dataset, batch_size=2, shuffle=True, num_workers=2)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Mask2FormerBinarySegmentation().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

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

            # Prepare Mask2Former targets
            mask_labels, class_labels = prepare_mask2former_targets(
                masks, num_queries=model.model.config.num_queries
            )

            # Forward
            outputs = model(
                pixel_values=images,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            # Loss is computed internally by Mask2Former
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate Dice score
            with torch.no_grad():
                predictions = model.predict(images, threshold=0.5)
                for i in range(predictions.shape[0]):
                    pred_binary = (predictions[i] > 0.5).float()
                    dice = dice_coefficient(pred_binary, masks[i])
                    epoch_dice_scores.append(dice)

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = sum(epoch_dice_scores) / len(epoch_dice_scores)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")

    # Save model
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'mask2former_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == '__main__':
    train()
