# 🧠 Resumen de Contexto para Agente Entrante — Chest X-Ray MLOps

> **Fecha de Generación:** 2026-04-13
> **Conversación Origen:** `8cc64dec-e08a-42b9-b15e-0cf0d4af6339`
> **Workspace:** `d:\2do4triMINAR\SL-XRay`

---

## 1. 🎯 Objetivo del Proyecto

Construir un **ecosistema comparativo multimodelo** que contraste el rendimiento de **8 algoritmos** (4 clásicos, 2 DL superficiales, 2 SOTA) contra un modelo CNN Desacoplado existente ("DeRC-X") del proyecto matriz ubicado en `d:\2do4triMINAR\Final-Algoritmos`. Todos los resultados deben confluir en una **única base de datos MLflow** para una comparativa científica unificada bajo la métrica `val_f1_score (Macro)`.

---

## 2. 📂 Arquitectura del Proyecto (Estado Actual)

```text
d:\2do4triMINAR\SL-XRay\
├── ML_Comparative_Plan.md          ← Blueprint/Plano Maestro (NO MODIFICAR)
├── CONTEXT_HANDOFF.md              ← Este archivo
├── chest_xray/                     ← Dataset crudo (fuente de verdad)
│   ├── COVID/       (138 imágenes, mixtas jpg/jpeg/png)
│   ├── NEUMONIA/    (1605 imágenes, mayoritariamente jpeg)
│   └── NORMALL/     (1399 imágenes, mayoritariamente jpeg)
└── Chest_XRay_Multimodel_Comparison/   ← Proyecto MLOps activo
    ├── data/                 ← Junction Link → d:\2do4triMINAR\SL-XRay\chest_xray
    ├── requirements_multimodel.txt
    ├── main.py               ← CLI Orquestador (--run classic|dl|sota|all)
    └── src/
        ├── data_pipeline.py          ← DualDataPipeline (PCA/Flatten para ML clásico, raw para HF)
        ├── classic_ml_pipeline.py    ← GaussianNB, RandomForest, KNN (GridSearch), SVM
        ├── deep_learning_pipeline.py ← MLP (Keras), Basic CNN (Keras)
        ├── hf_vision_pipeline.py     ← ResNet50 (HuggingFace), YOLOv8-cls (Ultralytics)
        └── metrics_evaluator.py      ← Registro centralizado a MLflow
```

---

## 3. 📊 Dataset: Distribución de Clases

| Clase | Cantidad | Índice (Label) |
|:---|:---:|:---:|
| **COVID** | 138 | `0` |
| **NEUMONIA** | 1,605 | `1` |
| **NORMALL** | 1,399 | `2` |
| **Total** | **3,142** | — |

> [!WARNING]
> **Desbalance severo.** COVID tiene ~10x menos muestras que las otras clases. Se aplica `SMOTE` para el pipeline clásico y `class_weight='balanced'` donde aplique.

---

## 4. 🔗 Integración MLflow (Crítico)

Todos los pipelines **DEBEN** registrar sus runs contra la base de datos del proyecto matriz:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///d:/2do4triMINAR/Final-Algoritmos/mlruns/mlflow.db")
mlflow.set_experiment("Chest_XRay_Senior_CNN")
```

### Tags obligatorios por familia:
| Familia | Tag `model_family` |
|:---|:---|
| Naive Bayes, RF, KNN, SVM | `Classic_ML` |
| MLP, Basic CNN | `Keras_DL` |
| ResNet50, YOLO | `HuggingFace` |

### Métrica Reina:
- **`val_f1_score`** → `f1_score(y_true, y_pred, average='macro')`

---

## 5. 🧩 Estado de Cada Módulo

### `src/data_pipeline.py` — ✅ Implementado
- Clase `DualDataPipeline` con 3 métodos de carga:
  - `get_classic_ml_data()`: Carga → Resize 64×64 → Flatten → SMOTE → PCA (n_components=50)
  - `get_deep_learning_data()`: Carga → Resize configurable → Normalización [0,1]
  - `get_hf_vision_data()`: Carga raw sin transformación (para inyectar en AutoImageProcessor/YOLO)

### `src/classic_ml_pipeline.py` — ✅ Implementado
- Clase `ClassicMLAlgorithms` con 4 entrenamientos:
  - `train_naive_bayes()` → GaussianNB
  - `train_random_forest()` → 100 estimadores, class_weight balanced
  - `train_knn()` → GridSearchCV sobre {n_neighbors: [3,5,7], weights: [uniform, distance]}
  - `train_svm()` → RBF kernel, class_weight balanced

### `src/deep_learning_pipeline.py` — ✅ Implementado
- Clase `DeepLearningAlgorithms`:
  - `build_mlp()` → Flatten → Dense(512) → Dense(256) → Softmax(3)
  - `build_basic_cnn()` → Conv2D(32) → Conv2D(64) → Dense(128) → Softmax(3)
  - `train_keras_model()` → Entrenamiento unificado con validation_split=0.2

### `src/hf_vision_pipeline.py` — ✅ Implementado (parcial)
- Clase `HFVisionAlgorithms`:
  - `get_resnet50()` → Carga de `microsoft/resnet-50` con `ignore_mismatched_sizes=True` para 3 clases
  - `train_yolo_cls()` → YOLOv8-cls con `ultralytics` (requiere `pip install ultralytics`)
- ⚠️ **No implementado aún:** Loop de fine-tuning de ResNet50 (solo carga el modelo) ni evaluación post-YOLO con métricas estandarizadas.

### `src/metrics_evaluator.py` — ✅ Implementado
- Clase `MetricsEvaluator` que:
  - Configura tracking_uri y experiment automáticamente
  - `evaluate_and_log()`: Calcula Macro-F1, logea params/tags/métricas, sube classification_report como artefacto

### `main.py` — ✅ Implementado
- CLI con `argparse`:
  - `--run classic` → Entrena NB, RF, KNN (sin SVM por ahora en el flujo principal)
  - `--run dl` → Entrena Basic CNN (1 epoch de smoke test)
  - `--run sota` → Solo carga ResNet50 en memoria (no entrena)
  - `--run all` → Los tres anteriores

---

## 6. ⚠️ Problemas Detectados / Pendientes

### Error en última ejecución:
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solución:** Ejecutar `pip install ultralytics` en el entorno virtual activo, o reorganizar los imports en `main.py` para que `hf_vision_pipeline` se importe condicionalmente.

### Problemas de Diseño a Corregir:
1. **Train/Test Split ausente:** `main.py` actualmente entrena y evalúa sobre el **mismo** set (X_train). Se necesita implementar `train_test_split` antes de evaluar para que las métricas sean válidas.
2. **SVM no incluido en el flujo `run_classic_ml()`** del `main.py` — el método existe en la clase pero no se invoca.
3. **MLP no incluido en `run_deep_learning()`** — solo se entrena Basic CNN.
4. **ResNet50 no tiene loop de fine-tuning** — solo carga el modelo pre-entrenado.
5. **YOLO-cls no se invoca** en `run_sota_vision()`.
6. **`temp_report.txt`** se escribe sin ruta absoluta, lo cual puede fallar dependiendo del CWD.

---

## 7. 📋 Portafolio de 8 Algoritmos (del Blueprint)

| # | Algoritmo | Pipeline | Estado |
|:---:|:---|:---|:---:|
| 1 | Naive Bayes | `classic_ml_pipeline.py` | ✅ |
| 2 | Random Forest | `classic_ml_pipeline.py` | ✅ |
| 3 | KNN (GridSearch) | `classic_ml_pipeline.py` | ✅ |
| 4 | SVM (RBF) | `classic_ml_pipeline.py` | ⚠️ Clase lista, no invocado |
| 5 | MLP (Keras) | `deep_learning_pipeline.py` | ⚠️ Build listo, no invocado |
| 6 | Basic CNN (Keras) | `deep_learning_pipeline.py` | ✅ |
| 7 | HF ResNet50 | `hf_vision_pipeline.py` | ⚠️ Solo carga, sin fine-tuning |
| 8 | YOLOv8-cls | `hf_vision_pipeline.py` | ⚠️ Método listo, no invocado |

---

## 8. 🛠️ Dependencias (`requirements_multimodel.txt`)

```
scikit-learn, imbalanced-learn, mlflow, transformers, datasets,
tensorflow, torch, torchvision, ultralytics, opencv-python,
pandas, numpy, matplotlib
```

---

## 9. 📁 Proyecto Matriz (Referencia)

El proyecto "padre" reside en `d:\2do4triMINAR\Final-Algoritmos\` y contiene:
- Su propio `src/config.py`, `src/data_pipeline.py`, `src/model_pipeline.py`
- La base de datos MLflow en `d:\2do4triMINAR\Final-Algoritmos\mlruns\mlflow.db`
- El dataset original compartido en `d:\2do4triMINAR\Final-Algoritmos\chest_xray` (misma fuente)

---

## 10. 🚀 Próximos Pasos para el Agente Entrante

1. **Instalar `ultralytics`** para desbloquear el import de `hf_vision_pipeline`.
2. **Implementar `train_test_split`** en `main.py` para que las métricas sean sobre datos de validación reales (no training set).
3. **Conectar los 4 algoritmos faltantes** (SVM, MLP, ResNet fine-tuning, YOLO-cls) al flujo del `main.py`.
4. **Ejecutar pasada completa** con `python main.py --run all` y verificar que los 8 runs aparezcan en `mlflow ui`.
5. **Generar reporte comparativo** Macro-F1 de los 8 algoritmos vs. el modelo DeRC-X del proyecto matriz.

---

## 11. 📌 Referencia Documental

| Documento | Ruta |
|:---|:---|
| Blueprint Maestro | `d:\2do4triMINAR\SL-XRay\ML_Comparative_Plan.md` |
| Orquestador | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\main.py` |
| Data Pipeline | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\src\data_pipeline.py` |
| ML Clásico | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\src\classic_ml_pipeline.py` |
| Deep Learning | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\src\deep_learning_pipeline.py` |
| HF/YOLO | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\src\hf_vision_pipeline.py` |
| Métricas MLflow | `d:\2do4triMINAR\SL-XRay\Chest_XRay_Multimodel_Comparison\src\metrics_evaluator.py` |
