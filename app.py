from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = "models/iris_rf.pkl"

app = FastAPI(title="Iris Model API")

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "API request count", ["endpoint"])
REQUEST_LATENCY = Summary("request_latency_seconds", "Request latency", ["endpoint"])

class PredictionRequest(BaseModel):
    inputs: list  # list of lists of features

model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
@REQUEST_LATENCY.labels(endpoint="/predict").time()
def predict(req: PredictionRequest):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    arr = np.array(req.inputs)
    preds = model.predict(arr).tolist()
    return {"predictions": preds}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
