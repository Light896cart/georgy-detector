import pandas as pd
import requests
from tqdm import tqdm
import os
from urllib.parse import urlparse


def simple_download_csv_images(csv_path, output_base_dir='images'):
    """Простое скачивание всех изображений из CSV"""

    # Читаем CSV
    df = pd.read_csv(csv_path)

    # Создаем базовую директорию
    os.makedirs(output_base_dir, exist_ok=True)

    success_count = 0
    error_count = 0

    # Проходим по всем строкам с прогресс-баром
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Скачивание"):
        url = row['image_path']
        label = row['label']

        # Создаем директорию для класса
        class_dir = os.path.join(output_base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        try:
            # Скачиваем изображение
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Определяем имя файла
            filename = url.split('/')[-1]
            if '?' in filename:
                filename = filename.split('?')[0]
            if not filename:
                filename = f"image_{idx}.jpg"

            # Сохраняем файл
            file_path = os.path.join(class_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)

            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"Ошибка при скачивании {url}: {e}")

    print(f"Успешно: {success_count}, Ошибок: {error_count}")

simple_download_csv_images(r'D:\Code\test_task_georges\data\combined_images.csv')