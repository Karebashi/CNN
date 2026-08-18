import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional until dependency is installed
    kagglehub = None


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
DEFAULT_DATASET_SLUG = 'bhavikjikadara/dog-and-cat-classification-dataset'


def _copy_images(source_dir: Path, target_dir: Path):
    if not source_dir.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for file in source_dir.iterdir():
        if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            shutil.copy2(file, target_dir / file.name)


def organize_dataset(dataset_root, destination_root=DATA_DIR):
    dataset_root = Path(dataset_root)
    destination_root = Path(destination_root)

    train_dir = destination_root / 'train'
    val_dir = destination_root / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for label in ('cat', 'dog'):
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (val_dir / label).mkdir(parents=True, exist_ok=True)

    # Caso 1: el dataset viene con train/val ya organizados
    for split_name, target_dir in [('train', train_dir), ('val', val_dir)]:
        source_split = dataset_root / split_name
        if source_split.exists():
            for label in ('cat', 'dog'):
                _copy_images(source_split / label, target_dir / label)
                alias_dir = source_split / (label + 's')
                if alias_dir.exists():
                    _copy_images(alias_dir, target_dir / label)

    # Caso 2: el dataset viene con carpetas directas cat/dog o cats/dogs
    for label in ('cat', 'dog'):
        for alias in (label, label + 's'):
            source_dir = dataset_root / alias
            if source_dir.exists() and list(source_dir.iterdir()):
                files = sorted(
                    p for p in source_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                )
                if not files:
                    continue

                split_index = max(1, int(len(files) * 0.8))
                for file in files[:split_index]:
                    shutil.copy2(file, train_dir / label / file.name)
                for file in files[split_index:]:
                    shutil.copy2(file, val_dir / label / file.name)

    return {'train': train_dir, 'val': val_dir}


def auto_download_if_needed(data_dir=DATA_DIR, dataset_slug=DEFAULT_DATASET_SLUG):
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    has_data = any((train_dir / label).exists() and any((train_dir / label).iterdir()) for label in ('cat', 'dog'))
    if not has_data:
        has_data = any((val_dir / label).exists() and any((val_dir / label).iterdir()) for label in ('cat', 'dog'))

    if has_data:
        return str(data_dir)

    if kagglehub is None:
        raise ImportError('Falta kagglehub. Instálalo con: pip install kagglehub')

    dataset_path = kagglehub.dataset_download(dataset_slug)
    organize_dataset(dataset_path, destination_root=data_dir)
    return str(dataset_path)


if __name__ == '__main__':
    result = auto_download_if_needed(DATA_DIR, DEFAULT_DATASET_SLUG)
    print(f'Dataset listo en: {result}')
