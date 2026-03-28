"""Скрипт для переупорядочивания датасета DVM-CAR, скачанного с kaggle.

Запускается только один раз. Для этого укажите путь скачанного датасета 
в SOURCE_DATASET. Поеле запуска все элементы будут упорядочены в папки 
по цветам. Также добавлено ограничение в 500 элементов для каждого цвета 
для балансировки данных.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def reorganize_dataset_by_color(source_dir, dest_dir):
    splits = {'train': 'train', 'validation': 'val', 'test': 'test'}

    # Создаем базовую папку для нового датасета
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for src_split, dst_split in splits.items():
        src_path = Path(source_dir) / src_split

        if not src_path.exists():
            print(f"Папка {src_path} не найдена. Пропускаем...")
            continue

        print(f"\nОбработка выборки: {src_split} -> {dst_split}...")
        
        all_files = []
        for root, _, files in os.walk(src_path):
            for file in files:
                all_files.append((Path(root), file))
                    
        # Идем по файлам и парсим цвета
        count = {}
        for root, file in tqdm(all_files, desc=f"Копирование {dst_split}"):
            # Шаблон: Brand$$Model$$Year$$Color$$...
            parts = file.split('$$')
            color = parts[3].strip()

            if color not in count:
                count[color] = 0

            if dst_split == 'train':
                if count[color] == 500:
                    continue
            elif count[color] == 100:
                continue

            count[color] += 1

            # Путь к новой папке цвета: dest_dir/val/White/
            color_dir = Path(dest_dir) / dst_split / color
            color_dir.mkdir(parents=True, exist_ok=True)
            
            old_file_path = root / file
            new_file_path = color_dir / file
            
            # Копируем картинку в новую папку цвета
            if not new_file_path.exists(): # На всякий случай, чтобы не перезаписывать зря
                shutil.copy2(old_file_path, new_file_path)

        for color, cnt in count.items():
            if dst_split == 'train' and cnt < 500 or \
               dst_split != 'train' and cnt < 100:
                dir = Path(dest_dir) / dst_split / color
                shutil.rmtree(dir)


if __name__ == "__main__":
    SOURCE_DATASET = ''
    NEW_DATASET = './src/dataset'

    reorganize_dataset_by_color(SOURCE_DATASET, NEW_DATASET)
    print("Готово! Датасет пересобран по цветам.")
