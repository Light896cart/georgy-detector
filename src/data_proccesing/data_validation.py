import pandas as pd
import os
from PIL import Image

total_path = r'D:\Code\test_task_georges\images'
csv_path = r'D:\Code\test_task_georges\data\combined_images_no_dupl.csv'
# Читаем CSV
df = pd.read_csv(csv_path)

print("Проверяю файлы...")

missing = 0
for idx, row in df.iterrows():
    img_url = row["image_path"]
    filename_with_ext = os.path.basename(img_url)
    label = row["label"]
    image_path = os.path.join(total_path, str(label), filename_with_ext)

    if not os.path.exists(image_path):
        print(f"[{idx}] Файл не найден: {image_path}")
        missing += 1
    else:
        try:
            Image.open(image_path).convert("RGB")  # Попробуем открыть
        except Exception as e:
            print(f"[{idx}] Ошибка при открытии: {image_path} | {e}")
            missing += 1

print(f"\nГотово. Проблемных файлов: {missing}")