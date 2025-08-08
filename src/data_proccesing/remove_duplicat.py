import pandas as pd

path = r'D:\Code\test_task_georges\data\combined_images.csv'
new_path = r'D:\Code\test_task_georges\data\combined_images_no_dupl.csv'
# Загрузить CSV файл
df = pd.read_csv(path)

# Удалить дубликаты
df_cleaned = df.drop_duplicates()

# Сохранить очищенный файл
df_cleaned.to_csv(new_path, index=False)

print(f"Было строк: {len(df)}")
print(f"Стало строк: {len(df_cleaned)}")
print(f"Удалено дубликатов: {len(df) - len(df_cleaned)}")