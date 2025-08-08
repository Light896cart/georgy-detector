import pandas as pd

# Загружаем файлы, указываем, что заголовков нет
georges = pd.read_csv(r'D:\Code\test_task_georges\data\georges.csv', header=None, names=['image_path'])
non_georges = pd.read_csv(r'D:\Code\test_task_georges\data\non_georges.csv', header=None, names=['image_path'])

# Добавляем метки классов
georges['label'] = 0
non_georges['label'] = 1

# Объединяем
combined = pd.concat([georges, non_georges], ignore_index=True)

# Сохраняем в новый CSV
combined.to_csv(r'D:\Code\test_task_georges\data\combined_images.csv', index=False)