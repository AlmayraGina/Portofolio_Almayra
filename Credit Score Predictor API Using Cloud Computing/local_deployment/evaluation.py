import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelEvaluator:
    """Handles pulling the pipeline back out from MLflow tracking and validating it."""
    
    def run(self, run_id: str, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
        print("--- Step 4: Evaluation ---")
        
        # Fetch the trained model/pipeline bundle from MLflow using run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        recall_per_class = recall_score(y_test,preds,average=None,labels=[0,1,2],zero_division=0)

        recall_0 = recall_per_class[0]

        # Log metric results to the active MLflow run context
        with mlflow.start_run(run_id=run_id):

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_class_0", recall_0)

        print(f"""Accuracy: {acc:.3f} Precision: {prec:.3f} Recall {rec:.3f} 
        Poor Creidt Scores Detected: {recall_0:.3f}""")

        return (acc, prec, recall_0)