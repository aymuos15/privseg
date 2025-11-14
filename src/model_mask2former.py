import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


class Mask2FormerBinarySegmentation(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-coco-instance"):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=2,  # Binary segmentation: background + lesion
            ignore_mismatched_sizes=True
        )
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass through Mask2Former

        Args:
            pixel_values: Input images [B, 3, H, W]
            mask_labels: Ground truth masks [B, num_queries, H, W] (optional, for training)
            class_labels: Ground truth class labels [B, num_queries] (optional, for training)
        """
        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        return outputs

    def predict(self, images, threshold=0.5):
        """
        Predict binary segmentation masks

        Args:
            images: Input images as PIL Images or tensors
            threshold: Confidence threshold for predictions

        Returns:
            Binary masks [B, 1, H, W]
        """
        # Process images
        if not isinstance(images, torch.Tensor):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.model.device)
        else:
            pixel_values = images

        # Get predictions
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        # Post-process to get binary masks
        # outputs.masks_queries_logits shape: [B, num_queries, H, W]
        # outputs.class_queries_logits shape: [B, num_queries, num_classes]

        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        binary_masks = []
        for i in range(batch_size):
            # Get class predictions for all queries
            class_probs = outputs.class_queries_logits[i].softmax(dim=-1)  # [num_queries, 2]
            lesion_probs = class_probs[:, 1]  # Probability of lesion class

            # Get mask logits and resize to input size
            mask_logits = outputs.masks_queries_logits[i]  # [num_queries, H_mask, W_mask]
            mask_probs = mask_logits.sigmoid()

            # Resize masks to match input image size
            mask_probs = torch.nn.functional.interpolate(
                mask_probs.unsqueeze(0),  # [1, num_queries, H_mask, W_mask]
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [num_queries, H, W]

            # Combine: weighted sum of masks by lesion probability
            final_mask = torch.zeros((h, w), device=pixel_values.device)
            for query_idx in range(mask_probs.shape[0]):
                if lesion_probs[query_idx] > threshold:
                    final_mask = torch.maximum(final_mask, mask_probs[query_idx])

            binary_masks.append(final_mask)

        # Stack and add channel dimension
        binary_masks = torch.stack(binary_masks).unsqueeze(1)  # [B, 1, H, W]
        return binary_masks
