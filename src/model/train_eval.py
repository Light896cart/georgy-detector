from typing import Optional, Dict, Any

import torch.optim as optim
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = None,
    load_weights_path: Optional[str] = "best_model.pth",  # откуда загружать (может быть None)
    save_best_path: str = "best_model_2.pth",  # куда сохранять лучшую
    verbose: bool = True,
    seed: int = 42,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    patience: int = 5,
) -> Dict[str, Any]:
    """
    Обучает модель с валидацией. Поддерживает загрузку весов из одного файла
    и сохранение лучшей модели в другой.

    Args:
        model: PyTorch модель
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        epochs: число эпох
        lr: learning rate
        device: 'cuda' или 'cpu', если None — автоматически
        load_weights_path: путь к весам для загрузки (None = начать с нуля)
        save_best_path: куда сохранить лучшую модель
        verbose: выводить ли прогресс
        seed: seed для воспроизводимости
        weight_decay: L2-регуляризация
        use_scheduler: использовать ли cosine annealing
        patience: остановка, если val_acc не растёт N эпох

    Returns:
        dict с метриками и моделью
    """
    # --- 🌱 Воспроизводимость ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 🖥️ Device setup ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- 🔁 Загрузка весов (если указан путь) ---
    if load_weights_path is not None:
        try:
            checkpoint = torch.load(load_weights_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if verbose:
                print(f"✅ Загружены веса из: {load_weights_path}")
        except FileNotFoundError:
            if verbose:
                print(f"⚠️ Веса не найдены: {load_weights_path}. Начинаем с нуля.")
        except Exception as e:
            print(f"❌ Ошибка при загрузке весов из {load_weights_path}: {e}")
    else:
        if verbose:
            print("🚀 Начинаем обучение с нуля (load_weights_path=None).")

    # --- ⚙️ Оптимизатор и лосс ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None

    # --- 📈 Логи ---
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    epochs_no_improve = 0

    # --- 🔁 Цикл обучения ---
    for epoch in range(epochs):
        # --- 🟠 Обучение ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            disable=not verbose,
            leave=False
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- 🟢 Валидация ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accuracies.append(val_acc)

        # --- 🏆 Сохраняем ЛУЧШУЮ модель в save_best_path ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_loss': epoch_loss,
                'model_config': {
                    'architecture': model.__class__.__name__,
                    'num_classes': model.fc.out_features,
                }
            }, save_best_path)
            if verbose:
                print(f"🎉 Новая лучшая модель сохранена в: {save_best_path} (acc={val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- 📢 Логирование ---
        if verbose:
            print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {epoch_loss:6.4f} | Val Acc: {val_acc:6.4f} | Best: {best_val_acc:6.4f}")

        # --- 🛑 Early stopping ---
        if patience is not None and epochs_no_improve >= patience:
            if verbose:
                print(f"🛑 Ранняя остановка на эпохе {epoch+1}")
            break

        # --- 🔄 LR Scheduler ---
        if scheduler is not None:
            scheduler.step()

    # --- 📈 Графики ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='tab:orange', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='tab:green', linewidth=2)
    plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.4f}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "model": model,
        "device": device,
    }