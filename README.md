# Heart Disease Risk Prediction: Production MLOps Framework
An enterprise-grade, reproducible end-to-end MLOps solution for heart disease risk classification utilizing the UCI Cleveland dataset. This repository automates data ingestion, model training, experiment tracking, continuous integration (CI/CD), multi-stage containerization, Kubernetes orchestration, and Prometheus-ready telemetry.

## System Architecture
The pipeline implements an isolated, reproducible design where every element, from training to serving, is versioned and validated.

```
       [ GitHub Push ] ---> [ CI/CD Action: Lint & Unit Tests ]
                                        │
                                        ▼
[ Production API Container ] <--- [ MLflow Registry ] <--- [ Automated Training ]
            │
            ▼
 [ Minikube / K8s Cluster ] ───> [ Prometheus / Grafana Telemetry ]
```

## Project Structure

```
heart-disease-mlops/
├── .github/workflows/    # CI/CD pipelines (GitHub Actions)
├── config/               # System configuration parameters
├── data/                 # Ingested and baseline data partitions (Git-ignored)
├── deployment/           # Kubernetes manifests (Deployment & LoadBalancer Service)
├── src/                  # Production Python source package
│   ├── app.py            # FastAPI Application & Prometheus metrics
│   ├── download_data.py  # Scripted data ingestion & structural cleaning
│   ├── eda.py            # Automated exploratory visual profiling
│   ├── pipelines.py      # Serializable Scikit-learn feature transformers
│   └── train.py          # Model execution & MLflow logging hooks
├── tests/                # Pytest validation test suites
├── Dockerfile            # Optimized multi-stage Docker build
├── requirements.txt      # Pinpointed dependency map
└── README.md             # Project documentation
```

## Quick Start & Local Setup
### 1. Prerequisites
Ensure you have the following runtimes installed locally:

Python 3.10

Docker Engine

kubectl + Minikube (for Kubernetes deployments)

### 2. Environment Initialization
Clone the repository and spin up a virtual environment:

```
git clone https://github.com/alphyelsa/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execution of Data & Training Pipelines
Trigger data acquisition, cleaning, and model experimentation:

```
# Fetch and clean the UCI Cleveland data
python src/download_data.py

# Run the exploratory data analysis suite
python src/eda.py

# Execute the training pipeline and log artifacts
python src/train.py
```

### 4. Viewing local Experimentation Logs
Launch the MLflow user interface to compare production model candidate runs:

```
# Linux / macOS : export MLFLOW_ALLOW_FILE_STORE=true
# Windows PowerShell: $env:MLFLOW_ALLOW_FILE_STORE="true"
set MLFLOW_ALLOW_FILE_STORE=true  #Windows cmd
mlflow ui --backend-store-uri ./src/mlruns --port 5000
```
Open your browser to http://localhost:5000 to inspect hyperparameters, evaluation metrics (Accuracy, Recall, ROC-AUC), and serialized model pipelines.

## Testing & Continuous Integration (CI/CD)
The code quality framework enforces rigorous checks before integration.

### 1. Local Unit Testing
To execute the comprehensive Pytest test suites across the features and serving modules:

```
pytest tests/ -v
```

### 2. Automated GitHub Actions Workflow
Every branch push or pull request to main fires an automated pipeline that:
- Provisions an isolated Ubuntu-latest environment.
- Formats and lints code via flake8.
- Validates functional integrity using pytest.
- Executes the full train.py pipeline to guarantee code execution stability.

### 3. Containerization & Local Serving
The inference layer uses a multi-stage Docker configuration to separate compilation dependencies from runtime environments, resulting in a minimal attack surface and small image footprint.

#### 3.1 Build the Image
```
docker build -t heart-disease-api:latest .
```
#### 3.2 Spin Up the Application
```
docker run -d -p 8000:8000 heart-disease-api:latest
```
#### 3.3 Verify Endpoints
Inference Endpoint: Send a POST payload to http://localhost:8000/predict
Swagger Documentation: Explore interface schema maps at http://localhost:8000/docs
Telemetry Exporter: Scrape raw Prometheus tracking vectors at http://localhost:8000/metrics

Sample Prediction Request Payload
```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0,
  "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0,
  "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
}'
```

### 4.Production Kubernetes Deployment
Deploy the stateless serving layer across a target cluster via core primitives.

#### 4.1. Initialize Minikube and Share the Docker Environment
```
minikube start
# Direct the local shell to use Minikube's Docker daemon
# eval $(minikube docker-env)
minikube docker-env --shell powershell | Invoke-Expression
docker info
# Re-build image directly inside the cluster daemon context
docker build -t heart-disease-api:latest .
```
#### 4.2. Apply Manifest Layouts
```
kubectl apply -f deployment/deployment.yaml
```
#### 4.3. Accessing the Service Application
Expose the cluster LoadBalancer to bind ingress traffic to your host network environment:
```
minikube service heart-disease-service
```

## Monitoring & Observability

The application exposes Prometheus metrics through the `/metrics` endpoint 
using `prometheus-fastapi-instrumentator`.

### 1. Install Prometheus and Grafana

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack
```

### 2. Deploy the ServiceMonitor

```bash
kubectl apply -f deployment/servicemonitor.yaml
```

### 3. Access Grafana

Start port forwarding:

```bash
kubectl port-forward svc/monitoring-grafana 3000:80
```

Open:

```
http://localhost:3000
```

Default username:

```
admin
```

Retrieve the admin password:

```bash
kubectl get secret monitoring-grafana \
-o jsonpath="{.data.admin-password}" | base64 --decode
```

### 4. Access Prometheus

```bash
kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
```

Open ip:

```
http://localhost:9090
```

### 5. Import Dashboard

1. Open Grafana.
2. Navigate to **Dashboards → Import**.
3. Upload `monitoring/grafana-dashboard.json`.
4. Select the Prometheus data source.
5. Save the dashboard.

The dashboard visualizes:
- API request rate
- Request latency
- Prediction endpoint traffic
- HTTP error rates