import os
import shutil
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def count_images(folder_path):
    if not os.path.isdir(folder_path):
        return 0
    return sum(
        1 for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith(IMAGE_EXTENSIONS)
    )


def copy_images(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(file, target_dir / file.name)


def prepare_dataset(dataset_root, base_dir='data'):
    dataset_root = Path(dataset_root)
    base_dir = Path(base_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f'No se encontró la carpeta del dataset: {dataset_root}')

    label_aliases = {
        'cat': ['cat', 'cats'],
        'dog': ['dog', 'dogs'],
    }

    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    ensure_dataset_structure(str(base_dir))

    # Caso A: el dataset ya trae train/val con carpetas cat/dog
    for split_name, split_dir in [('train', train_dir), ('val', val_dir)]:
        source_split = dataset_root / split_name
        if source_split.exists():
            for label, aliases in label_aliases.items():
                for alias in aliases:
                    source_label_dir = source_split / alias
                    if source_label_dir.exists():
                        copy_images(source_label_dir, split_dir / label)

    # Caso B: el dataset tiene carpetas directas cat/dog o cats/dogs
    for label, aliases in label_aliases.items():
        for alias in aliases:
            source_dir = dataset_root / alias
            if source_dir.exists():
                total = count_images(source_dir)
                if total > 0:
                    train_target = train_dir / label
                    val_target = val_dir / label
                    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                    if not files:
                        continue
                    split_index = max(1, int(len(files) * 0.8))
                    for file in files[:split_index]:
                        shutil.copy2(file, train_target / file.name)
                    for file in files[split_index:]:
                        shutil.copy2(file, val_target / file.name)

    return {
        'train': train_dir,
        'val': val_dir,
    }


def ensure_dataset_structure(base_dir):
    required = [
        os.path.join(base_dir, 'train', 'cat'),
        os.path.join(base_dir, 'train', 'dog'),
        os.path.join(base_dir, 'val', 'cat'),
        os.path.join(base_dir, 'val', 'dog'),
    ]

    for path in required:
        os.makedirs(path, exist_ok=True)

    return required


def validate_dataset(train_dir, val_dir):
    required = {
        'train_cat': os.path.join(train_dir, 'cat'),
        'train_dog': os.path.join(train_dir, 'dog'),
        'val_cat': os.path.join(val_dir, 'cat'),
        'val_dog': os.path.join(val_dir, 'dog'),
    }

    counts = {name: count_images(path) for name, path in required.items()}

    if counts['train_cat'] == 0 or counts['train_dog'] == 0:
        raise ValueError(
            'No hay imágenes suficientes en train. Debes poner al menos 1 imagen de cada clase en:\n'
            f"- {required['train_cat']}\n- {required['train_dog']}\n"
            'Formato esperado: archivos .jpg, .jpeg, .png, .bmp o .webp.'
        )

    if counts['val_cat'] == 0 or counts['val_dog'] == 0:
        raise ValueError(
            'No hay imágenes suficientes en val. Debes poner al menos 1 imagen de cada clase en:\n'
            f"- {required['val_cat']}\n- {required['val_dog']}\n"
            'Formato esperado: archivos .jpg, .jpeg, .png, .bmp o .webp.'
        )

    return counts


def build_generators(train_dir, val_dir, image_size=(128, 128), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    if train_generator.samples == 0 or val_generator.samples == 0:
        raise ValueError(
            'Los generadores de imagen no tienen ejemplos. Verifica que las carpetas contengan imágenes reales.'
        )

    return train_generator, val_generator
