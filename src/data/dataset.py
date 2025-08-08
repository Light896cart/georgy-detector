import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urlparse
import time
from typing import Optional, Callable, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Data_Georges(Dataset):
    """
    Dataset для бинарной классификации Георгия Победоносца.

    Работает в двух режимах:
    1. Если total_path задан → ищет изображения локально: {total_path}/{label}/{image_path}
    2. Если total_path=None → считает, что image_path в CSV — это URL, грузит онлайн.
    """

    def __init__(
            self,
            csv_path: str,
            total_path: Optional[str] = None,  # если None — режим URL
            transform: Optional[Callable] = None,
            cache_images: bool = False,
            max_retries: int = 3,
            timeout: int = 5,
    ):
        self.csv_path = csv_path
        self.total_path = total_path
        self.transform = transform
        self.cache_images = cache_images
        self.max_retries = max_retries
        self.timeout = timeout

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Проверяем обязательные колонки
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV должен содержать колонки 'image_path' и 'label'")

        # Кэш
        self._cache = {}

        # Логика: локальный режим или URL
        if self.total_path is not None:
            logger.info(f"Режим: локальные файлы (папка: {self.total_path})")
            # Проверим, что папка существует
            if not os.path.exists(self.total_path):
                raise FileNotFoundError(f"Папка с изображениями не найдена: {self.total_path}")
        else:
            logger.info("Режим: загрузка по URL")

    def _load_image_local(self, img_path: str, label: int) -> Image.Image:
        """Загружает изображение из локальной папки."""
        filename = os.path.basename(img_path)
        local_path = os.path.join(self.total_path, str(label), filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Файл не найден: {local_path}")
        return Image.open(local_path).convert("RGB")

    def _load_image_url(self, url: str) -> Image.Image:
        """Загружает изображение по URL с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Не удалось загрузить изображение по URL: {url}") from e
                # Лёгкая задержка
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError("Unreachable")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)}).")

        try:
            row = self.df.iloc[idx]
            img_path = row["image_path"].strip()  # на всякий случай
            label = int(row["label"])

            # Кэш: если уже загружали
            if self.cache_images and idx in self._cache:
                image = self._cache[idx]
            else:
                # Определяем режим по total_path
                if self.total_path is not None:
                    image = self._load_image_local(img_path, label)
                else:
                    image = self._load_image_url(img_path)

                # Сохраняем в кэш (копию, чтобы transform не портил)
                if self.cache_images:
                    self._cache[idx] = image.copy()

            # Применяем трансформации
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logger.error(f"Ошибка при загрузке элемента {idx}: {e}")
            raise e

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)