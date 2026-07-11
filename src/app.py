import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Risk Inference Service")

# Instrument API with Prometheus Metrics hooks
Instrumentator().instrument(app).expose(app)

# Load production model artifact safely (local directory or remote MLflow Registry)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/mlruns/0/")
# For production standalone simplicity, ensure model is copied or packaged.
try:
    # Point directly to your serialized MLflow artifact directory or tracking URI
    model_run_dirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
    latest_run = model_run_dirs[-1]
    model = mlflow.sklearn.load_model(f"{MODEL_PATH}/{latest_run}/artifacts/model")
except Exception as e:
    model = None
    print(f"Warning: Model payload could not be pre-loaded: {e}")


class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    # Convert input payload to DataFrame matching feature naming convention
    input_df = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(probabilities[prediction])

    return {
        "heart_disease_risk": prediction,
        "confidence": confidence
    }