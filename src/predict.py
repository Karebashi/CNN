import os
import sys

from tensorflow.keras.preprocessing import image

from model import build_cnn

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pet_classifier.keras')


def predict_single_image(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'No se encontró el modelo en {MODEL_PATH}. Entrena primero con: python src/train.py')

    model = build_cnn(input_shape=(128, 128, 3), num_classes=1)
    model.load_weights(MODEL_PATH)

    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))

    probability = float(model.predict(img_array, verbose=0)[0][0])
    label = 'dog' if probability >= 0.5 else 'cat'

    print(f'Predicción: {label}')
    print(f'Probabilidad de perro: {probability:.4f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Uso: python src/predict.py "ruta/a/la/imagen.jpg"')
        raise SystemExit(1)

    predict_single_image(sys.argv[1])
