import tensorflow as tf
from tensorflow.keras import layers, models

class DeepLearningAlgorithms:
    """Provisión local de redes MLP y arquitecturas CNN Base (Baseline)."""
    def __init__(self, input_shape=(128, 128, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_mlp(self):
        """Red Perceptrón Multicapa puro sobre imagen aplanada."""
        model = models.Sequential([
            layers.Flatten(input_shape=self.input_shape),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, {"model": "MLP_Keras", "layers": "512-256", "dropout": 0.5}

    def build_basic_cnn(self):
        """CNN Básica que servirá como piso comparativo absoluto."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, {"model": "Basic_CNN", "filters": "32-64"}
    
    def train_keras_model(self, model, X_train, y_train, epochs=10, batch_size=32):
        print("Entrenando modelo de Deep Learning en TensorFlow...")
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2, 
            verbose=1
        )
        return model, history
