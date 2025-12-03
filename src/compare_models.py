import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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


def evaluate_model(model, test_loader, device, model_name="Model", store_predictions=False):
    """Evaluate a model on test set"""
    model.eval()
    dice_scores = []
    iou_scores = []

    # Storage for visualization
    predictions_list = []
    images_list = []
    masks_list = []

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

            # Store for visualization
            if store_predictions:
                predictions_list.append(predictions.cpu())
                images_list.append(images.cpu())
                masks_list.append(masks.cpu())

    results = {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'dice_scores': dice_scores,
        'iou_scores': iou_scores
    }

    if store_predictions:
        results['predictions'] = torch.cat(predictions_list, dim=0)
        results['images'] = torch.cat(images_list, dim=0)
        results['masks'] = torch.cat(masks_list, dim=0)

    return results


def plot_metrics_comparison(unet_results, mask2former_results, save_dir):
    """Create bar charts and box plots comparing metrics"""
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Bar chart for mean metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Dice Score', 'IoU Score']
    unet_means = [unet_results['dice_mean'], unet_results['iou_mean']]
    m2f_means = [mask2former_results['dice_mean'], mask2former_results['iou_mean']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, unet_means, width, label='U-Net', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, m2f_means, width, label='Mask2Former', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Mean Metric Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Box plot for Dice scores
    ax2 = fig.add_subplot(gs[0, 1])
    dice_data = [unet_results['dice_scores'], mask2former_results['dice_scores']]
    bp1 = ax2.boxplot(dice_data, labels=['U-Net', 'Mask2Former'],
                      patch_artist=True, showmeans=True)
    bp1['boxes'][0].set_facecolor('#3498db')
    bp1['boxes'][1].set_facecolor('#e74c3c')
    for box in bp1['boxes']:
        box.set_alpha(0.6)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    # Box plot for IoU scores
    ax3 = fig.add_subplot(gs[0, 2])
    iou_data = [unet_results['iou_scores'], mask2former_results['iou_scores']]
    bp2 = ax3.boxplot(iou_data, labels=['U-Net', 'Mask2Former'],
                      patch_artist=True, showmeans=True)
    bp2['boxes'][0].set_facecolor('#3498db')
    bp2['boxes'][1].set_facecolor('#e74c3c')
    for box in bp2['boxes']:
        box.set_alpha(0.6)
    ax3.set_ylabel('IoU Score', fontsize=12)
    ax3.set_title('IoU Score Distribution', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                           indices, category_name, save_dir):
    """Plot sample predictions for both models"""
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 5, figsize=(18, 3.5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Original image
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image', fontsize=10)
        axes[i, 0].axis('off')

        # Ground truth
        gt = masks[idx, 0].numpy()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')

        # UNet prediction
        unet_pred = unet_preds[idx, 0].numpy()
        axes[i, 2].imshow(unet_pred, cmap='gray')
        axes[i, 2].set_title('U-Net Prediction', fontsize=10)
        axes[i, 2].axis('off')

        # Mask2Former prediction
        m2f_pred = m2f_preds[idx, 0].numpy()
        axes[i, 3].imshow(m2f_pred, cmap='gray')
        axes[i, 3].set_title('Mask2Former Prediction', fontsize=10)
        axes[i, 3].axis('off')

        # Error comparison overlay
        axes[i, 4].imshow(img)
        # UNet errors in blue, Mask2Former errors in red, agreement in green
        unet_correct = (unet_pred == gt).astype(float)
        m2f_correct = (m2f_pred == gt).astype(float)

        # Create RGB overlay
        overlay = np.zeros((*gt.shape, 3))
        overlay[:, :, 1] = (unet_correct * m2f_correct) * gt  # Green: both correct
        overlay[:, :, 0] = (m2f_correct * (1 - unet_correct)) * gt  # Red: only M2F correct
        overlay[:, :, 2] = (unet_correct * (1 - m2f_correct)) * gt  # Blue: only UNet correct

        axes[i, 4].imshow(overlay, alpha=0.5)
        axes[i, 4].set_title('Agreement (G=Both, R=M2F, B=UNet)', fontsize=9)
        axes[i, 4].axis('off')

    plt.suptitle(f'{category_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'samples_{category_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_analysis(images, masks, unet_preds, m2f_preds,
                       unet_scores, m2f_scores, save_dir):
    """Create error analysis visualizations"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Per-sample performance scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(unet_scores, m2f_scores, alpha=0.6, s=50, color='#3498db')
    ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Equal Performance')
    ax1.set_xlabel('U-Net Dice Score', fontsize=12)
    ax1.set_ylabel('Mask2Former Dice Score', fontsize=12)
    ax1.set_title('Per-Sample Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Performance difference histogram
    ax2 = fig.add_subplot(gs[0, 1])
    diff = np.array(m2f_scores) - np.array(unet_scores)
    ax2.hist(diff, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax2.set_xlabel('Dice Difference (Mask2Former - U-Net)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Performance Difference Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Find worst case (lowest average score)
    avg_scores = [(unet_scores[i] + m2f_scores[i]) / 2 for i in range(len(unet_scores))]
    worst_idx = np.argmin(avg_scores)

    # Worst case visualization
    ax3 = fig.add_subplot(gs[1, 0])
    img = images[worst_idx].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Create error overlay
    gt = masks[worst_idx, 0].numpy()
    unet_pred = unet_preds[worst_idx, 0].numpy()
    m2f_pred = m2f_preds[worst_idx, 0].numpy()

    # Red: False Positive, Blue: False Negative, Green: True Positive
    error_map = np.zeros((*gt.shape, 3))
    error_map[:, :, 1] = gt * unet_pred  # Green: True positive
    error_map[:, :, 0] = (1 - gt) * unet_pred  # Red: False positive
    error_map[:, :, 2] = gt * (1 - unet_pred)  # Blue: False negative

    ax3.imshow(img)
    ax3.imshow(error_map, alpha=0.5)
    ax3.set_title(f'Worst Case - U-Net (Dice: {unet_scores[worst_idx]:.3f})\nG=TP, R=FP, B=FN',
                  fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    error_map_m2f = np.zeros((*gt.shape, 3))
    error_map_m2f[:, :, 1] = gt * m2f_pred
    error_map_m2f[:, :, 0] = (1 - gt) * m2f_pred
    error_map_m2f[:, :, 2] = gt * (1 - m2f_pred)

    ax4.imshow(img)
    ax4.imshow(error_map_m2f, alpha=0.5)
    ax4.set_title(f'Worst Case - Mask2Former (Dice: {m2f_scores[worst_idx]:.3f})\nG=TP, R=FP, B=FN',
                  fontsize=11, fontweight='bold')
    ax4.axis('off')

    plt.savefig(os.path.join(save_dir, 'error_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_visualizations(unet_results, mask2former_results, save_dir):
    """Create all visualizations for model comparison"""
    os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. Metrics comparison
    print("  - Creating metrics comparison plots...")
    plot_metrics_comparison(unet_results, mask2former_results, save_dir)

    # Extract data
    images = unet_results['images']
    masks = unet_results['masks']
    unet_preds = unet_results['predictions']
    m2f_preds = mask2former_results['predictions']
    unet_dice_scores = unet_results['dice_scores']
    m2f_dice_scores = mask2former_results['dice_scores']

    # 2. Sample predictions for different categories
    n_samples = len(images)

    # Best cases (high scores for both)
    avg_scores = [(unet_dice_scores[i] + m2f_dice_scores[i]) / 2 for i in range(n_samples)]
    best_indices = np.argsort(avg_scores)[-5:][::-1]
    print("  - Creating best case visualizations...")
    plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                          best_indices, "Best Cases (Highest Average Dice)", save_dir)

    # Worst cases (low scores for both)
    worst_indices = np.argsort(avg_scores)[:5]
    print("  - Creating worst case visualizations...")
    plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                          worst_indices, "Worst Cases (Lowest Average Dice)", save_dir)

    # UNet wins
    diff_scores = np.array(unet_dice_scores) - np.array(m2f_dice_scores)
    unet_win_indices = np.argsort(diff_scores)[-5:][::-1]
    print("  - Creating U-Net advantage cases...")
    plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                          unet_win_indices, "U-Net Performs Better", save_dir)

    # Mask2Former wins
    m2f_win_indices = np.argsort(diff_scores)[:5]
    print("  - Creating Mask2Former advantage cases...")
    plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                          m2f_win_indices, "Mask2Former Performs Better", save_dir)

    # Highest disagreement
    disagreement = []
    for i in range(n_samples):
        unet_p = unet_preds[i, 0].numpy()
        m2f_p = m2f_preds[i, 0].numpy()
        disagree_ratio = np.abs(unet_p - m2f_p).sum() / unet_p.size
        disagreement.append(disagree_ratio)

    disagree_indices = np.argsort(disagreement)[-5:][::-1]
    print("  - Creating high disagreement cases...")
    plot_sample_predictions(images, masks, unet_preds, m2f_preds,
                          disagree_indices, "Highest Model Disagreement", save_dir)

    # 3. Error analysis
    print("  - Creating error analysis plots...")
    plot_error_analysis(images, masks, unet_preds, m2f_preds,
                       unet_dice_scores, m2f_dice_scores, save_dir)

    print(f"\n✓ All visualizations saved to: {save_dir}/")


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

    unet_results = evaluate_model(unet, test_loader, device, "U-Net", store_predictions=True)
    mask2former_results = evaluate_model(mask2former, test_loader, device, "Mask2Former", store_predictions=True)

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

    # Generate visualizations
    viz_dir = os.path.join(project_root, 'results/comparison_visualizations')
    create_visualizations(unet_results, mask2former_results, viz_dir)


if __name__ == '__main__':
    compare()
