# 🚀 MLOps Chest X-Ray Multimodel Classification System

Plataforma MLOps *End-to-End* diseñada para clasificar radiografías de tórax (`COVID`, `NEUMONIA`, `NORMALL`) mediante un ecosistema comparativo de **8 algoritmos** que abarca modelos clásicos (SVM, KNN, Naive Bayes, Random Forest), redes neuronales superficiales (MLP, CNN Básica) y modelos SOTA de visión (ResNet50 HuggingFace, YOLOv8-cls Ultralytics). La tubería implementa ingeniería de datos bifurcada (SMOTE + PCA para ML clásico, tensores nativos para Deep Learning) y converge todos los experimentos en un servidor centralizado de observabilidad MLflow bajo la métrica unificada `val_f1_score (Macro)`.

## 📂 Estructura del Nodo del Proyecto

- `Chest_XRay_Multimodel_Comparison/src/`: Tuberías ETL bifurcadas (`DualDataPipeline`), módulos algorítmicos por familia y evaluador de métricas centralizado.
- `Chest_XRay_Multimodel_Comparison/main.py`: Orquestador CLI unificado para ejecutar subconjuntos del portafolio (`--run classic|dl|sota|all`).
- `Chest_XRay_Multimodel_Comparison/Final_Experimentation_Report.md`: Reporte académico de 9 secciones con resultados comparativos, interpretabilidad y arquitectura MLOps.
- `chest_xray/`: Dataset de radiografías de tórax (3,142 imágenes en 3 clases). *Excluido del repositorio por peso binario; descargar por separado.*

---

## 🛠️ 1. Instalación y Configuración Inicial

### Clonar el repositorio y acceder a la terminal:
```bash
git clone https://github.com/Many871027/ML-DL-X-Ray-Clasifications-MLOps.git
cd ML-DL-X-Ray-Clasifications-MLOps
```

### Instaurar el Entorno Virtual Aislado:
Aislar el entorno de trabajo asegura reproducibilidad entre distintos computadores (Windows PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Sincronización de Dependencias:
```powershell
pip install -r Chest_XRay_Multimodel_Comparison/requirements_multimodel.txt
```
*(Asegúrate de que el dataset `chest_xray/` con las subcarpetas `COVID/`, `NEUMONIA/` y `NORMALL/` se encuentre anclado en la raíz del proyecto. El enlace simbólico `data/` dentro de `Chest_XRay_Multimodel_Comparison/` apunta automáticamente a esta fuente).*

---

## 🔬 2. Entrenamiento y Sistema de Experimentación

Nuestra arquitectura MLOps compite automáticamente **8 arquitecturas algorítmicas** utilizando validación estratificada con `train_test_split(stratify=y)`, SMOTE para balanceo de la clase minoritaria COVID, y PCA para reducción dimensional.

**Opción A: Ejecución Completa del Portafolio (Recomendado)**
```powershell
.\venv\Scripts\python Chest_XRay_Multimodel_Comparison/main.py --run all
```

**Opción B: Ejecución por Familia de Modelos**
```powershell
# Solo modelos clásicos (NB, RF, KNN, SVM)
.\venv\Scripts\python Chest_XRay_Multimodel_Comparison/main.py --run classic

# Solo Deep Learning superficial (MLP, Basic CNN)
.\venv\Scripts\python Chest_XRay_Multimodel_Comparison/main.py --run dl

# Solo modelos SOTA (ResNet50, YOLOv8-cls)
.\venv\Scripts\python Chest_XRay_Multimodel_Comparison/main.py --run sota
```

### Observabilidad (MLflow Tracking Server)
Una vez desencadenada la experimentación, todas las métricas (`val_f1_score`), parámetros, tags de familia (`Classic_ML`, `Keras_DL`, `HuggingFace`) y reportes de clasificación habrán sido interceptados. Revisa analíticamente tu interfaz ejecutando:
```powershell
.\venv\Scripts\mlflow ui --backend-store-uri sqlite:///d:/2do4triMINAR/Final-Algoritmos/mlruns/mlflow.db
```
🌐 Navega a **[http://localhost:5000](http://localhost:5000)** y selecciona el experimento `Chest_XRay_Senior_CNN`.

---

## 📊 3. Resultados del Benchmark Comparativo

| Ranking | Algoritmo | Familia | Macro F1-Score (Val) |
|:-------:|:----------|:--------|:--------------------:|
| 🏆 **1** | **SVM (RBF)** | Classic ML | **0.9479** |
| **2** | Basic CNN | Keras DL | 0.9356 |
| **3** | Random Forest | Classic ML | 0.9289 |
| **4** | KNN (GridSearch) | Classic ML | 0.9255 |
| **5** | MLP (Keras) | Keras DL | 0.8491 |
| **6** | Naive Bayes | Classic ML | 0.8217 |
| **7** | YOLOv8-cls | HF/SOTA | 0.8000 |
| **8** | ResNet50 (Base) | HF/SOTA | 0.3060 |

> **Hallazgo clave:** El SVM (RBF) superó a las arquitecturas de Deep Learning, demostrando que la *ingeniería de datos* (SMOTE + PCA) tiene más impacto que la complejidad arquitectónica del modelo cuando el volumen de datos es limitado ($N < 5,000$).

---

## ⚙️ 4. Pipeline de Inferencia para Producción

El modelo ganador (SVM) opera bajo el siguiente flujo de inferencia:

```
Imagen Raw (JPEG/PNG) → Resize (64×64) → Flatten → PCA (n=50) → SVM.predict() → {COVID, NEUMONIA, NORMALL}
```

Para una documentación técnica completa del ecosistema, consulta el reporte en:
📄 `Chest_XRay_Multimodel_Comparison/Final_Experimentation_Report.md`
