import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

class ResizeKeepAR:
    """
    Resize с сохранением пропорций + паддинг до целевого размера
    """
    def __init__(self, target_size=512, fill=0):
        self.target_size = target_size
        self.fill = fill  # цвет паддинга (0 = чёрный)

    def __call__(self, img):
        w, h = img.size

        # Масштабируем, сохраняя AR
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Центрируем на чёрном фоне
        new_img = Image.new("RGB", (self.target_size, self.target_size), (self.fill, self.fill, self.fill))
        left = (self.target_size - new_w) // 2
        top = (self.target_size - new_h) // 2
        new_img.paste(img, (left, top))

        return new_img

# Теперь используем с torchvision
def get_transforms():
    train_transform = transforms.Compose([
        ResizeKeepAR(target_size=150, fill=0),  # Сохраняем AR + паддинг
        transforms.RandomCrop(150, pad_if_needed=True, padding_mode='constant', fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(10, interpolation=F.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 🔽 Валидация: только resize + центральный кроп + нормализация
    val_transform = transforms.Compose([
        ResizeKeepAR(target_size=150, fill=0),  # То же самое
        transforms.CenterCrop(150),              # ✅ Только центр, без случайности
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform