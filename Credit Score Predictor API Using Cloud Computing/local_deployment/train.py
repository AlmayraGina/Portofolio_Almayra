from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
from preprocessing import CreditScorePreprocssor

class ModelConfig:

    CLASS_WEIGHTS = {2:0.8, 1: 2.0, 0: 2.8}

    SMOTE_STRATEGY = {0: 25000,1: 14000,2: 18000}

class ModelFactory:

    @staticmethod
    def decision_tree():
        return DecisionTreeClassifier(max_depth=10, min_samples_leaf=2,class_weight=ModelConfig.CLASS_WEIGHTS)

    @staticmethod
    def random_forest():
        return RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=4,class_weight=ModelConfig.CLASS_WEIGHTS)

    @staticmethod
    def xgboost():

        return XGBClassifier(n_estimators=100,max_depth=10,learning_rate=0.01,subsample=0.5,
                             colsample_bytree=0.8,objective="multi:softprob",num_class=3,eval_metric="mlogloss")

    @staticmethod
    def get_all_models():
        return {
            "Decision Tree": ModelFactory.decision_tree(),
            "Random Forest": ModelFactory.random_forest(),
            "XGBoost": ModelFactory.xgboost()
        }

class CreditScoreTrainer:

    def __init__(self, experiment_name="Credit Score Prediction", artifact_path="artifacts"):

        self.preprocessor = CreditScorePreprocssor()
        self.models = ModelFactory.get_all_models()
        self.smote = SMOTE(sampling_strategy=ModelConfig.SMOTE_STRATEGY)
        self.artifact_dir = Path(artifact_path)
        self.artifact_dir.mkdir(parents=True,exist_ok=True)

        mlflow.set_experiment(experiment_name)

    def run(self, data_path):

        X_train, X_test, y_train, y_test = (self.preprocessor.clean_and_split(data_path))
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        X_train, y_train = self.smote.fit_resample(X_train,y_train)

        feature_names = X_train.columns.tolist()

        best_model = None
        best_run_id = None
        best_priority_recall=0

        for name, model in self.models.items():

            with mlflow.start_run(
                run_name=name
            ) as run:

                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test,y_pred)
                precision = precision_score(y_test,y_pred,average="macro",zero_division=0)
                recall = recall_score(y_test,y_pred,average="macro",zero_division=0)
                f1 = f1_score(y_test,y_pred,average="macro")

                mlflow.log_metric("accuracy",acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("f1_score", f1)

                mlflow.log_param("model_type", name)

                mlflow.sklearn.log_model(model,name="model")

                recall_per_class = recall_score(y_test, y_pred,average=None, labels=[0,1,2],zero_division=0)
                priority_recall = recall_per_class[0]

                print(f"{name} recall for poor credit scores: {priority_recall}")

                if priority_recall > best_priority_recall:
                    best_priority_recall = priority_recall
                    best_model = model
                    best_run_id = run.info.run_id

        model_path = (self.artifact_dir /"best_model.pkl")

        joblib.dump({"preprocessor": self.preprocessor,"model": best_model,"feature_names": feature_names},model_path)
        print(f"Best prediction model is saved to {model_path}")

        return (best_run_id,X_test,y_test)

    def apply_smote(self, X_train, y_train):

        X_smote, y_smote = self.smote.fit_resample(X_train,y_train)

        return X_smote, y_smote

    def train(self, X_train, y_train):

        for name, model in self.models.items():
            print(f"{name} is training")
            model.fit(X_train,y_train)