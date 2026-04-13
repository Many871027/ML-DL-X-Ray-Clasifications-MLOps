# 🚀 Plan de Referencia: Ecosistema Comparativo Multimodelo (Chest X-Ray)

**Instrucción para el Agente de la Nueva Ventana:**
> *Este documento es el **Plano Maestro (Blueprint)**. Si eres el agente Antigravity de la nueva sesión, tu objetivo es ejecutar estrictamente paso a paso la arquitectura descrita aquí, manteniendo un enfoque local y registrando los experimentos hacia la base de datos MLflow del proyecto matriz.*

---

## 1. 🎯 Objetivo del Proyecto
Construir una tubería de validación científica que contraste el rendimiento matemático del modelo "DeRC-X" (CNN Desacoplada Matriz) contra un portafolio expansivo de algoritmos clásicos, redes superficiales y modelos de visión del estado del arte (SOTA) descargados desde `Hugging Face`.

## 2. 📂 Arquitectura del Directorio (Scaffolding Local)
El nuevo espacio de trabajo debe aislarse, pero compartir la fuente de datos:

```text
/Chest_XRay_Multimodel_Comparison/
├── data/
│   └── (Symlink o acceso a D:\2do4triMINAR\Final-Algoritmos\chest_xray)
├── notebooks/
│   ├── 01_feature_extraction_classic_ml.ipynb
│   ├── 02_classic_ml_training.ipynb
│   └── 03_huggingface_finetuning.ipynb
├── src/
│   ├── classic_ml_pipeline.py  # KNN, SVM, Navie Bayes, Random Forest
│   ├── deep_learning_pipeline.py # MLP, Basic CNN
│   ├── hf_vision_pipeline.py   # ResNet (HF), YOLO
│   └── metrics_evaluator.py    # Reporte estandarizado Macro-F1
└── main.py
```

## 3. 🧠 Estrategia de Ingeniería de Datos (Bifurcación Geométrica)
Dado que los distintos algoritmos procesan el espacio de manera diferente, el agente debe construir un Dataloader Dual:

### A. Para Modelos Clásicos (SVM, KNN, NB, RF) y MLP:
- **Problema:** No entienden matrices 2D/3D (Imágenes).
- **Procesamiento Requerido:** 
  1. Redimensionamiento a una escala muy pequeña (Ej. `64x64`) para no desbordar RAM.
  2. Aplanamiento vectorial (Flatten) a tensores 1D.
  3. **Extracción de Características (Obligatorio/Senior):** Utilizar Análisis de Componentes Principales (`PCA`) para reducir dimensionalidad o `HOG` (Histogram of Oriented Gradients) para que el SVM o KNN extraiga verdaderos contornos pulmonares en lugar de pixeles vacíos.

### B. Para Hugging Face (ResNet, YOLO):
- **Problema:** Requieren transformaciones pre-estructuradas.
- **Procesamiento Requerido:**
  1. Utilizar `AutoImageProcessor.from_pretrained("microsoft/resnet-50")`.
  2. Implementar los tensores nativos de PyTorch (`pt`) o TensorFlow (`tf`) exigidos por la API *Transformers*.
  
> [!WARNING]
> No aplicar `Medical-Safe RandomTranslation` a los validadores de HuggingFace ya que ellos usarán el *Feature Extractor* oficial congelado.

## 4. 🧮 Implementación de Algoritmos (El Portafolio)

| Familia de Modelos | Algoritmos Específicos | Consideraciones de Implementación Local |
| :--- | :--- | :--- |
| **Estadísticos / Clásicos** | **1.** Naive Bayes <br> **2.** Random Forest | Ejecutar en CPU (`scikit-learn`). Rápidos pero altamente susceptibles a la "ilusión de precisión" del desbalance. Necesitan métrica ponderada. |
| **Basados en Distancia** | **3.** KNN <br> **4.** SVM | Extremadamente costosos en memoria RAM si no se aplica `PCA`. Usar un *GridSearch* nativo pequeño para KNN (`n_neighbors`). |
| **Deep Learning Superficial**| **5.** MLP <br> **6.** Basic CNN | Usar `Keras` / TensorFlow. MLP sufrirá si la imagen es > 128x128. Las CNN estándar servirán como baseline absoluto frente al proyecto matriz. |
| **Transfer Learning SOTA** | **7.** HF ResNet50 <br> **8.** HF YOLO (v8/v11) | Ocuparán intensamente los 3GB de VRAM local. Obligatorio prender el `TF_FORCE_GPU_ALLOW_GROWTH` o forzar el entrenamiento por `PyTorch Accelerator`. YOLO requiere reformateo de Labels a Bounding Boxes vacíos o conversión a clasificación. |

## 5. 🔬 MLOps: Estrategia de Rastrero Comparativo (Crucial)
El verdadero poder de este proyecto residirá en que **todos los puntos converjan**.

Para que la comparativa tenga sentido científico contra la iteración actual:
1. El agente no debe crear un log de MLflow aislado.
2. Debe apuntar el rastreador *exactamente a la misma base de datos*:
   ```python
   import mlflow
   mlflow.set_tracking_uri("sqlite:///d:/2do4triMINAR/Final-Algoritmos/mlruns/mlflow.db")
   mlflow.set_experiment("Chest_XRay_Senior_CNN")
   ```
3. A cada corrida clásica o pre-entrenada se le asignará la etiqueta (Tag): `{"model_family": "Classic_ML"}` o `{"model_family": "HuggingFace"}`.
4. **La Métrica Reina:** Todos los algoritmos, sin excepción, deberán reportar localmente su desempeño usando `val_f1_score (Macro)` para que queden codo a codo registrados contra nuestro modelo CNN actual.

---
💡 **Al nuevo Agente Antigravity:** Comprende profundamente los entornos virtuales. Inicia creando un archivo `requirements_multimodel.txt` que involucre `scikit-learn`, `transformers`, e `imbalanced-learn`.
