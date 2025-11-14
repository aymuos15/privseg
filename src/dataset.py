import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class HAM10000Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_ids, transform=None, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load mask
        mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
        mask = Image.open(mask_path).convert('L')
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Binarize

        return image, mask
