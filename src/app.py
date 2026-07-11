import os
import time
import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from prometheus_fastapi_instrumentator import Instrumentator

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Risk Inference Service")

# Instrument API with Prometheus Metrics hooks
Instrumentator().instrument(app).expose(app)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "Request completed | method=%s path=%s status=%s duration=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response


# Load production model artifact
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")

try:
    model = mlflow.sklearn.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully")

except Exception as e:
    model = None
    print(f"Model loading failed: {e}")


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
    request_start = time.time()

    if model is None:
        logger.error("Prediction request failed: model unavailable")
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    try:
        # Convert input payload to DataFrame
        preprocessing_start = time.time()

        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        preprocessing_time = time.time() - preprocessing_start

        # Model inference timing
        inference_start = time.time()

        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(input_df)[0]

        confidence = float(
            probabilities[prediction]
        )

        inference_time = time.time() - inference_start

        total_time = time.time() - request_start

        logger.info(
            "Prediction completed | prediction=%s confidence=%.4f "
            "preprocessing_time=%.4fs inference_time=%.4fs total_time=%.4fs",
            prediction,
            confidence,
            preprocessing_time,
            inference_time,
            total_time
        )

        return {
            "heart_disease_risk": prediction,
            "confidence": confidence
        }


    except Exception as e:

        logger.exception(
            "Prediction failed | error=%s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )
