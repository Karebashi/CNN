import os
import threading
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / 'models' / 'pet_classifier.keras'


def preprocess_image(image_path: str, target_size=(128, 128)):
    image = Image.open(image_path).convert('RGB').resize(target_size)
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


class PetClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Clasificador de perros y gatos')
        self.root.geometry('760x520')
        self.root.minsize(640, 420)

        self.model = None
        self.current_image_path = None

        self.frame = tk.Frame(root, padx=18, pady=18)
        self.frame.pack(fill='both', expand=True)

        self.header = tk.Label(
            self.frame,
            text='Clasificador de imágenes',
            font=('Arial', 18, 'bold')
        )
        self.header.pack(pady=(0, 12))

        controls = tk.Frame(self.frame)
        controls.pack(fill='x', pady=(0, 12))

        self.select_button = tk.Button(
            controls,
            text='Seleccionar imagen',
            command=self.select_image,
            width=20,
            height=2,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold')
        )
        self.select_button.pack(side='left', padx=(0, 10))

        self.predict_button = tk.Button(
            controls,
            text='Clasificar',
            command=self.predict,
            width=18,
            height=2,
            bg='#2196F3',
            fg='white',
            font=('Arial', 11, 'bold'),
            state='disabled'
        )
        self.predict_button.pack(side='left')

        self.status_var = tk.StringVar(value='Cargando modelo...')
        self.status_label = tk.Label(self.frame, textvariable=self.status_var, fg='#333333', font=('Arial', 10))
        self.status_label.pack(anchor='w', pady=(0, 6))

        center = tk.Frame(self.frame)
        center.pack(fill='both', expand=True)

        preview_frame = tk.LabelFrame(center, text='Vista previa', padx=10, pady=10)
        preview_frame.pack(side='left', fill='both', expand=True, padx=(0, 12))

        self.preview_label = tk.Label(preview_frame, text='Sin imagen seleccionada', width=38, height=18, bg='#f2f2f2', fg='#555555', compound='center')
        self.preview_label.pack(fill='both', expand=True)

        result_frame = tk.LabelFrame(center, text='Resultado', padx=12, pady=12)
        result_frame.pack(side='right', fill='both', expand=True)

        self.result_var = tk.StringVar(value='Esperando imagen...')
        self.result_label = tk.Label(
            result_frame,
            textvariable=self.result_var,
            font=('Arial', 16, 'bold'),
            wraplength=240,
            justify='center',
            fg='#222222'
        )
        self.result_label.pack(fill='both', expand=True)

        self.root.after(100, self.load_model)

    def load_model(self):
        def worker():
            try:
                from tensorflow import keras

                if not MODEL_PATH.exists():
                    self.root.after(0, lambda: self.status_var.set('Modelo no encontrado. Entrena primero con: python src/train.py'))
                    self.root.after(0, lambda: self.predict_button.config(state='disabled'))
                    return

                self.model = keras.models.load_model(str(MODEL_PATH))
                self.root.after(0, lambda: self.status_var.set('Modelo cargado correctamente.'))
            except Exception as exc:  # pragma: no cover - UI path only
                self.root.after(0, lambda: self.status_var.set(f'Error al cargar el modelo: {exc}'))
                self.root.after(0, lambda: self.predict_button.config(state='disabled'))

        threading.Thread(target=worker, daemon=True).start()

    def select_image(self):
        if self.model is None:
            messagebox.showwarning('Modelo no disponible', 'No se pudo cargar el modelo. Primero entrena el modelo.')
            return

        filetypes = [
            ('Imágenes', '*.png;*.jpg;*.jpeg;*.bmp;*.webp'),
            ('Todos los archivos', '*.*')
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return

        self.current_image_path = path
        self.result_var.set('Imagen cargada. Listo para clasificar.')
        self.predict_button.config(state='normal')

        try:
            image = Image.open(path).convert('RGB')
            image.thumbnail((360, 360))
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo, text='')
            self.preview_label.image = photo
        except Exception as exc:
            messagebox.showerror('Error', f'No se pudo abrir la imagen: {exc}')

    def predict(self):
        if self.model is None or not self.current_image_path:
            messagebox.showwarning('Sin imagen', 'Primero selecciona una imagen o espera a que el modelo cargue.')
            return

        try:
            array = preprocess_image(self.current_image_path)
            probability = float(self.model.predict(array, verbose=0)[0][0])
            if probability >= 0.5:
                label = 'Perro'
                info = f'Perro ({probability * 100:.1f}%)'
            else:
                label = 'Gato'
                info = f'Gato ({(1 - probability) * 100:.1f}%)'

            self.result_var.set(info)
            self.status_var.set(f'Predicción: {label}')
        except Exception as exc:
            messagebox.showerror('Error de predicción', f'No se pudo clasificar la imagen: {exc}')


def main():
    root = tk.Tk()
    PetClassifierGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
