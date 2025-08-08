import torch
from matplotlib import pyplot as plt

from src.data.dataloader import create_train_val_dataloaders
from src.data.dataset import Data_Georges
from src.model.model_architecture import replace_basicblock_with_se, SELayer, create_model
from src.model.train_eval import train_and_evaluate_model
from src.utils.seeding import set_seed
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils


import torchvision.models as models
from torchvision.models import ResNet18_Weights


def main(img_dir_path=None, csv_path=None, dataset=None):
    # 1. Воспроизводимость
    print("🔧 Устанавливаем seed для воспроизводимости...")
    set_seed(42)

    print("CSV путь:", csv_path)
    print("\n📥 Загружаем и разбиваем данные...")
    train_loader, val_loader = create_train_val_dataloaders(
        csv_path=csv_path,
        img_dir_path=img_dir_path,
        dataset=dataset,
        train_ratio=0.01,
        fraction=0.20,  # 100% данных
        batch_size=20,
        num_workers=0,
        seed=42
    )

    # 🧠 Создаём и настраиваем модель
    model = create_model(pretrained=True, freeze_backbone=True, reduction=8)

    # 🚀 Обучение
    train_and_evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1
    )

if __name__ == "__main__":

    img_dir_path = r'D:\Code\test_task_georges\images'
    csv_path = r'D:\Code\test_task_georges\data\combined_images_no_dupl.csv'
    dataset = Data_Georges
    main(img_dir_path=img_dir_path,csv_path=csv_path,dataset=dataset)