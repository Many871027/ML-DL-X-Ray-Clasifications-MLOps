import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from transformers import AutoImageProcessor

class DualDataPipeline:
    def __init__(self, data_dir="data", img_size_classic=(64, 64)):
        self.data_dir = Path(data_dir)
        self.img_size_classic = img_size_classic
        self.classes = ['COVID', 'NEUMONIA', 'NORMALL'] # Mapeado a las carpetas

        # Inicialización aplazada para procesadores SOTA pesados
        self._hf_processor = None

    def _load_raw_images(self, resize=None):
        """Carga iterativa de imágenes y etiquetas desde el directorio"""
        images = []
        labels = []
        for label_idx, class_name in enumerate(self.classes):
            class_path = self.data_dir / class_name
            if not class_path.exists():
                print(f"Warning: Directorio no encontrado {class_path}")
                continue

            print(f"Cargando clase: {class_name}...")
            for img_name in os.listdir(class_path):
                img_path = class_path / img_name
                if not img_path.is_file(): continue
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if resize:
                    img = cv2.resize(img, resize)

                images.append(img)
                labels.append(label_idx)

        if resize:
            return np.array(images), np.array(labels)
        else:
            # Retornamos lista para evitar ValueError de NumPy con formas inhomogéneas
            return images, np.array(labels)

    def get_classic_ml_data(self, use_pca=True, n_components=100, apply_smote=True):
        """Vectores 1D (Flatten) para el portafolio Clásico de ML."""
        print(f"--- PREPARANDO DATOS ML CLÁSICO ({self.img_size_classic}) ---")
        X, y = self._load_raw_images(resize=self.img_size_classic)

        X_flat = X.reshape(len(X), -1)

        if apply_smote:
            print("Aplicando SMOTE para balanceo de clases...")
            sm = SMOTE(random_state=42)
            X_flat, y = sm.fit_resample(X_flat, y)

        if use_pca:
            print(f"Aplicando PCA (Componentes={n_components})...")
            pca = PCA(n_components=n_components, random_state=42)
            X_flat = pca.fit_transform(X_flat)
            print("PCA Finalizado.")
            return X_flat, y, pca

        return X_flat, y, None

    def get_deep_learning_data(self, img_size=(128, 128)):
        """Tensor base escalado para Keras MLP / Basic CNN"""
        print(f"--- PREPARANDO DATOS KERAS/TF ({img_size}) ---")
        X, y = self._load_raw_images(resize=img_size)
        X = X.astype('float32') / 255.0
        return X, y

    def get_hf_vision_data(self):
        """Retorna rutas crudas o tensores compatibles con HF Pipelines y YOLO"""
        print("--- PREPARANDO DATOS HUGGING FACE / SOTA ---")
        # En ecosistemas SOTA, la carga por lotes es crítica por VRAM.
        # Retornaremos la representación en crudo validada para inyectar en constructores de Dataset
        # YOLO usa directorios crudos, ResNet usa AutoImageProcessor por lotes.
        return self._load_raw_images()
