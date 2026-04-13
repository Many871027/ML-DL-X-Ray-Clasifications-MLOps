import torch
from transformers import ResNetForImageClassification
from ultralytics import YOLO

class HFVisionAlgorithms:
    """Provisión local de Modelos Hugging Face (VQA, Image Classifiers) y YOLO."""
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_resnet50(self):
        """Retorna modelo Transformers para fine-tuning/evaluación."""
        print("Cargando microsoft/resnet-50 desde HuggingFace Hub...")
        model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        model.to(self.device)
        return model, {"model": "HF_ResNet50"}

    def train_yolo_cls(self, data_path, epochs=10, imgsz=128):
        """
        Entrenamiento YOLOv8 Nano de Clasificación.
        YOLO requiere que la carpeta esté organizada en clases.
        """
        print("Iniciando Pipeline YOLO-cls (Classification)...")
        model = YOLO('yolov8n-cls.pt') 
        
        print(f"Lanzando entrenamiento YOLO en la ruta {data_path}...")
        
        # El motor de YOLO maneja internamente DataLoader, Augmentations y Logeo.
        results = model.train(
            data=data_path, 
            epochs=epochs, 
            imgsz=imgsz, 
            device='0' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        
        return model, {"model": "YOLOv8-cls", "epochs": epochs, "imgsz": imgsz}
