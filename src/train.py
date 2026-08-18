import argparse
import os

from model import build_cnn
from data_utils import build_generators, ensure_dataset_structure, validate_dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pet_classifier.keras')


def main():
    parser = argparse.ArgumentParser(description='Entrena un clasificador de perros y gatos.')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print('Usando las imágenes ya presentes en data/train y data/val')
    ensure_dataset_structure(DATA_DIR)
    validate_dataset(
        train_dir=os.path.join(DATA_DIR, 'train'),
        val_dir=os.path.join(DATA_DIR, 'val'),
    )

    train_generator, val_generator = build_generators(
        train_dir=os.path.join(DATA_DIR, 'train'),
        val_dir=os.path.join(DATA_DIR, 'val'),
        image_size=(128, 128),
        batch_size=32,
    )

    model = build_cnn(input_shape=(128, 128, 3), num_classes=1)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
    print('Training finished.')


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        print(f'Error de datos: {exc}')
        raise SystemExit(1)
