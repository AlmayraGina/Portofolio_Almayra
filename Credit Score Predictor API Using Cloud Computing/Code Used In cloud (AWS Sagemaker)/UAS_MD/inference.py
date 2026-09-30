import json
import os
import joblib
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CLASS_NAMES = {0: "Poor",1: "Standard",2: "Good"}

def model_fn(model_dir):

    model_path = os.path.join(model_dir, "best_model.pkl")

    artifact = joblib.load(model_path)

    return artifact


def input_fn(request_body, request_content_type):

    if request_content_type != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    if "instances" not in payload:
        raise ValueError("JSON must contain 'instances'")

    return pd.DataFrame(payload["instances"])


def predict_fn(input_data, artifact):

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]

    processed_data = preprocessor.transform(input_data)
    processed_data = processed_data[model.feature_names_in_]

    predictions = model.predict(processed_data)

    response = {
        "predictions": predictions.tolist(),
        "labels": [CLASS_NAMES[int(p)] for p in predictions]
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_data)
        response["probabilities"] = probabilities.tolist()

    return response


def output_fn(prediction, accept):

    return (json.dumps(prediction),JSON_CONTENT_TYPE)