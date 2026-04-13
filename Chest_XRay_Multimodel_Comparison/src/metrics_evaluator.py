import mlflow
from sklearn.metrics import f1_score, classification_report
import time

class MetricsEvaluator:
    def __init__(self, experiment_name="Chest_XRay_Senior_CNN", tracking_uri="sqlite:///d:/2do4triMINAR/Final-Algoritmos/mlruns/mlflow.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"[MLOps] Tracking apuntado a: {self.tracking_uri} | Experimento: {self.experiment_name}")

    def evaluate_and_log(self, run_name, model_family, y_true, y_pred, params=None):
        """
        Calcula F1-Score (Macro) y registra todo en el experimento central.
        """
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        with mlflow.start_run(run_name=run_name):
            # Tags
            mlflow.set_tag("model_family", model_family)
            mlflow.set_tag("timestamp", str(time.time()))
            
            # Params
            if params:
                mlflow.log_params(params)
                
            # Métrica Reina
            mlflow.log_metric("val_f1_score", macro_f1)
            
            # Log verbose de classification report como artefacto si se desea en el futuro
            report = classification_report(y_true, y_pred, target_names=['COVID', 'NEUMONIA', 'NORMALL'])
            print(f"--- Evaluación {run_name} ({model_family}) ---")
            print(f"Macro F1-Score: {macro_f1:.4f}")
            print(report)
            
            # Save report to local temp file to log as artifact
            with open("temp_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("temp_report.txt")
            
        return macro_f1
