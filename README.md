# CNN para detectar perros y gatos enjaulados

Este proyecto crea una red neuronal convolucional (CNN) para clasificar una imagen en dos clases:
- `cat`
- `dog`

La idea es entrenar el modelo con imágenes de perros y gatos, y luego usarlo para predecir una imagen nueva.

## Estructura del proyecto

- `src/train.py`: entrena el modelo
- `src/predict.py`: predice la clase de una imagen
- `src/model.py`: define la CNN
- `src/data_utils.py`: carga y prepara los datos
- `models/`: guarda el modelo entrenado

## Organización esperada de datos

Coloca tus imágenes así:

```text
data/
  train/
    cat/
      cat_001.jpg
      cat_002.jpg
    dog/
      dog_001.jpg
      dog_002.jpg
  val/
    cat/
      cat_001.jpg
    dog/
      dog_001.jpg
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenamiento

```bash
python src/train.py
```

## Predicción

```bash
python src/predict.py "ruta/a/tu/imagen.jpg"
```

## Resultado esperado

La salida puede ser algo como:

```text
Predicción: dog
Probabilidad: 0.94
```

> Nota: para imágenes con jaulas, la CNN puede aprender mejor si se incluye una variedad de fotos con fondos y perspectivas diferentes.
