import os
import torch
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union

from src.model.model_architecture import create_model

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Классы ---
CLASS_NAMES = {0: "Это Георгий Победоносец", 1: "Не Георгий Победоносец"}

def load_model(weights_path: str = 'best_model.pth', device: str = None) -> torch.nn.Module:
    """
    Загружает обученную модель.
    Предполагается, что модель — это ResNet-18 с SE-блоками.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}. "
                                f"Ожидается: {os.path.abspath(weights_path)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(pretrained=False, freeze_backbone=False)  # как в train.py
    model.to(device)
    model.eval()

    # Загружаем веса
    checkpoint = torch.load(weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"✅ Модель загружена из: {weights_path}")
    return model

def load_image(image_source: Union[str, Path]) -> Image.Image:
    """
    Загружает изображение:
    - если строка начинается с http → грузит с URL
    - иначе → считает путём к локальному файлу
    """
    if isinstance(image_source, (Path, str)) and str(image_source).startswith(('http://', 'https://')):
        response = requests.get(str(image_source), timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image_path = Path(image_source)
        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        image = Image.open(image_path).convert("RGB")
    return image

def predict_image(model: torch.nn.Module, image_source: Union[str, Path], device: str = None) -> Tuple[str, float]:
    """
    Предсказывает один образ.
    Args:
        model: обученная модель
        image_source: путь к файлу или URL
        device: 'cpu' или 'cuda'

    Returns:
        (класс, вероятность)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image = load_image(image_source)
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            class_idx = pred.item()
            conf = confidence.item()

        class_name = CLASS_NAMES[class_idx]
        return class_name, conf

    except Exception as e:
        return f"Ошибка: {str(e)}", 0.0

def predict_from_folder(model: torch.nn.Module, folder_path: str, device: str = None):
    """
    Предсказывает для всех изображений в папке.
    Поддерживаемые форматы: .jpg, .jpeg, .png, .bmp, .tiff
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Папка не найдена: {folder}")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"❌ Нет изображений в папке: {folder}")
        return

    print(f"\n🔍 Анализируем {len(image_files)} изображений из папки: {folder}")
    results = []

    for img_path in image_files:
        class_name, conf = predict_image(model, img_path, device)
        results.append((img_path.name, class_name, conf))
        print(f"🖼️  {img_path.name:30} → {class_name} (вероятность: {conf:.4f})")

    return results

def visualize_predictions(image_source: Union[str, Path], prediction: str, confidence: float):
    """
    Показывает изображение и результат предсказания.
    """
    try:
        image = load_image(image_source)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"{prediction}\n(вероятность: {confidence:.4f})", fontsize=14, color='green' if "Георгий" in prediction else 'red')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ Не удалось визуализировать: {e}")


# --- CLI — можно запускать как скрипт ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🔍 Инференс модели: Георгий Победоносец или нет",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python predict.py --weights best_model.pth --image test.jpg
  python predict.py --weights best_model.pth --image "https://example.com/georgy.jpg"
  python predict.py --weights best_model.pth --folder ./images/
  python predict.py --weights best_model.pth  # ← попросит выбрать image или folder
        """
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pth",
        help="Путь к файлу весов (по умолчанию: best_model.pth)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Путь к изображению или URL"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Путь к папке с изображениями"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Устройство (по умолчанию: auto)"
    )

    args = parser.parse_args()

    # --- 🔍 Проверка весов ---
    if not os.path.exists(args.weights):
        print(f"❌ Файл с весами не найден: {args.weights}")
        print(f"💡 Убедитесь, что файл лежит в: {os.path.abspath(args.weights)}")
        exit(1)

    # --- 🔍 Проверка: что загружать ---
    if not args.image and not args.folder:
        print("🤔 Не указано, что анализировать.")
        print("👉 Укажите одно из:")
        print("   --image <путь_или_URL>    → для одного изображения")
        print("   --folder <папка>          → для нескольких изображений")
        print("\nПримеры:")
        print("   python predict.py --weights best_model.pth --image test.jpg")
        print("   python predict.py --weights best_model.pth --folder ./images/")
        print("\nИли используйте --help для справки.")
        exit(1)

    # --- ✅ Всё ок — продолжаем ---
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    if args.image:
        pred, conf = predict_image(model, args.image, device)
        print(f"\n🎯 Результат: {pred} (вероятность: {conf:.4f})")
        visualize_predictions(args.image, pred, conf)

    if args.folder:
        predict_from_folder(model, args.folder, device)