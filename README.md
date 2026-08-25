# Cats vs Dogs — MLOps Assignment 2

An end-to-end MLOps implementation for binary Cats vs Dogs image classification. This project covers the complete machine learning lifecycle, bridging the gap between local model development and production-grade deployment:

* **Dataset Management:** Tracked and versioned using [DVC](https://dvc.org/).
* **Model Training & Evaluation:** A custom PyTorch Convolutional Neural Network ([`CatsDogsCNN`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/src/model.py#L7)).
* **Automated Testing:** Thorough test suite implemented with `pytest`.
* **Containerization:** Production API built using [Docker](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/Dockerfile).
* **Continuous Integration (CI):** Automates tests, checks model integrity, and pushes images to the **GitHub Container Registry (GHCR)** via [GitHub Actions CI](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/.github/workflows/ci.yml).
* **Continuous Deployment (CD):** Deploys the container to a local **Kind** Kubernetes cluster using [GitHub Actions CD](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/.github/workflows/cd.yml).
* **Local Image Registry:** Facilitates image transfer into Kind using a local Docker registry.
* **Kubernetes Health Checks:** Readiness and liveness probes targeting FastAPI `/health`.
* **Deployment Validation:** Post-deployment smoke tests that verify API behavior on container startup.
* **Observability & Evaluation:** Real-time request metrics via `/metrics` and automated post-deployment model evaluation.

---

## 1. Project Overview

The objective of this assignment is to build and deploy a production-style machine learning inference service for binary image classification. The trained model classifies input images as either `cat` or `dog`, and is served via a FastAPI application. The final production deployment runs inside a local Kubernetes cluster created using **Kind** (Kubernetes in Docker).

---

## 2. MLOps Architecture

The project's architectural components represent a complete, closed-loop machine learning lifecycle. The block diagram below illustrates the sequential flow from training data to production monitoring:

```text
                         ┌─────────────────────┐
                         │   Cats vs Dogs Data │
                         │       + DVC         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Training      │
                         │ CatsDogsCNN          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ best_model.pt       │
                         │ Production Artifact │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FastAPI Application  │
                         │ /health             │
                         │ /predict            │
                         │ /metrics            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Automated Tests     │
                         │ pytest              │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Docker Build        │
                         │ Linux / AMD64       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ GHCR                │
                         │ SHA + latest tags   │
                         └──────────┬──────────┘
                                    │
                           CI successful
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ GitHub Actions CD   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Local Registry      │
                         │ localhost:5001      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Kind Kubernetes     │
                         │ cats-dogs-mlops     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cats vs Dogs API    │
                         │ Deployment + Service│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Smoke Tests         │
                         │ Health + Prediction │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Monitoring          │
                         │ /metrics + logs     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Post-Deployment     │
                         │ Evaluation          │
                         └─────────────────────┘
```

---

## 3. End-to-End Workflow Demonstration

The complete MLOps workflow demonstrated by this project is:

```text
Developer changes code
        │
        ▼
Git push to main
        │
        ▼
GitHub Actions CI
        │
        ├── Automated tests
        ├── Model validation
        ├── Docker build
        └── GHCR push
                │
                ▼
        CI successful
                │
                ▼
        GitHub Actions CD
                │
                ├── Pull exact SHA image
                ├── Push to local registry
                ├── Deploy to Kind
                ├── Wait for rollout
                ├── Health check
                └── Prediction smoke test
                        │
                        ▼
                  Deployed API
                        │
                ┌───────┴────────┐
                ▼                ▼
             /metrics       Evaluation
                │                │
                ▼                ▼
          Runtime metrics   Model performance
```

This provides an end-to-end implementation of CI, CD, deployment validation, monitoring, and post-deployment model evaluation for the Cats vs Dogs inference service.

---

## 4. Repository Structure

Below is the directory structure of the repository, showing the organization of the API, training scripts, testing suite, Kubernetes manifests, and CI/CD workflows:

```text
BITS-MTech-MLOps-Assignment-2/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── src/
│   ├── __init__.py
│   ├── create_splits.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── inspect_dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── scripts/
│   ├── __init__.py
│   ├── check_dataset_integrity.py
│   ├── create_test_model.py
│   ├── post_deployment_evaluation.py
│   ├── smoke_test.py
│   ├── test_data_pipeline.py
│   ├── test_inference.py
│   └── test_model_gpu.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── test_api.py
│   ├── test_dataset.py
│   └── test_model.py
│
├── artifacts/
│   └── best_model.pt
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── .dvc/
├── data/
│   ├── processed.dvc
│   └── raw/
│       └── PetImages.dvc
│
├── Dockerfile
├── .dockerignore
├── .dvcignore
├── .gitignore
├── params.yaml
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

* **Configuration:** The local Kind registry configuration is maintained in [`kind-registry.yaml`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/kind-registry.yaml).
* **Runner Environment:** Note that the local GitHub Actions runner installation is environment-specific and is not included in the repository submission.

---

## 5. Dataset Management

We manage the Cats vs Dogs dataset using **Data Version Control (DVC)** to keep the repository lightweight and version-controlled.
* **Raw Dataset Representation:** Tracked via the DVC file [`data/raw/PetImages.dvc`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/data/raw/PetImages.dvc). 
* **Git vs DVC:** The actual raw images are excluded from Git due to their size. DVC tracks the raw images and dataset state, while Git tracks the corresponding metadata.
* **Classes:** The dataset is split into two classes under the folders `Cat/` and `Dog/`.

---

## 6. Model Architecture & Serving

This project employs a custom convolutional neural network, [`CatsDogsCNN`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/src/model.py#L7), for binary classification. 

* **Production Model Artifact:** The trained production model weights are saved at [`artifacts/best_model.pt`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/artifacts/best_model.pt).
* **Serving Strategy:** The FastAPI application loads this model directly on startup.
* **Hardware & Runtime:** The inference service runs on the CPU. The model is baked directly into the Docker image, ensuring the container does not need to fetch weights from external storage at runtime, which speeds up startup and avoids network failures.

---

## 7. FastAPI Inference Service

The inference service is implemented in [`api/main.py`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/api/main.py) using **FastAPI** and exposes three main endpoints:

### Health Check
* **Endpoint:** `GET /health`
* **Purpose:** Used by Kubernetes readiness/liveness probes, CI/CD deployment validation, and smoke tests to confirm the service and model are ready.
* **Example Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model": "CatsDogsCNN",
    "device": "cpu"
  }
  ```

### Prediction
* **Endpoint:** `POST /predict`
* **Purpose:** Accepts an image via multipart form upload and returns the predicted class along with its probability.
* **Supported Formats:** JPEG, PNG, WebP (invalid file types are rejected).
* **Example Request:**
  ```bash
  curl.exe -X POST "http://localhost:8001/predict" \
    -F "file=@tests/fixtures/cat.jpg"
  ```
* **Example Response:**
  ```json
  {
    "predicted_label": "cat",
    "class_probabilities": {
      "cat": 0.6641,
      "dog": 0.3359
    },
    "confidence": 0.6641
  }
  ```

### Metrics
* **Endpoint:** `GET /metrics`
* **Purpose:** Exposes real-time, in-process operational metrics.
* **Example Response:**
  ```json
  {
    "total_requests": 30,
    "successful_requests": 29,
    "failed_requests": 0,
    "prediction_requests": 23,
    "cat_predictions": 10,
    "dog_predictions": 13,
    "average_latency_ms": 55.43
  }
  ```

---

## 8. Testing

We use `pytest` for automated unit and integration tests. Run the full test suite using:
```bash
pytest -v
```

The test suite verifies:
* **API Endpoints:** Health checks, prediction correctness for cats and dogs, and edge cases (e.g., empty or invalid files).
* **Data Pipelines:** Dataset classes and input sample shapes.
* **Model Integration:** Model output shapes and predictions.

---

## 9. Docker Containerization

The application is packaged into a production container using the [`Dockerfile`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/Dockerfile).
* **Image Contents:** Includes the FastAPI server, Python dependencies, inference code, and the baked-in production model weights.
* **Platform Support:** Built specifically for `linux/amd64`.
* **Metadata Attestation:** The CI build pipeline disables Docker buildx provenance/attestation metadata. This ensures that the generated image can be seamlessly loaded and moved between GHCR, Docker Desktop, the local registry, and the local Kind cluster without any compatibility issues.

---

## 10. GitHub Container Registry (GHCR)

Our Docker images are hosted on GHCR at:
`ghcr.io/varpeprasanna/bits-mtech-mlops-assignment-2`

Every successful CI build publishes two tags:
1. `:<FULL_GIT_SHA>` (Immutable tag matching the exact source commit)
2. `:latest` (Mutable tag pointing to the most recent successful build)

We use the immutable Git SHA tag during deployment to guarantee that the CD pipeline deploys the exact image built and verified by CI.

---

## 11. Continuous Integration (CI)

CI is automated via the GitHub Actions workflow defined in [`.github/workflows/ci.yml`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/.github/workflows/ci.yml).
* **Triggers:** Automatically runs on every push or pull request to the `main` branch.
* **Pipeline Steps:**
  1. Checks out the repository and configures Python.
  2. Installs required dependencies.
  3. Runs the automated `pytest` suite.
  4. Validates the production model file integrity using a SHA256 checksum comparison.
  5. Builds the Docker image.
  6. Logs into GHCR and pushes the Docker image tagged with both the commit SHA and `latest`.

---

## 12. Continuous Deployment (CD)

CD is automated via the GitHub Actions workflow defined in [`.github/workflows/cd.yml`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/.github/workflows/cd.yml).
* **Triggers:** Automatically runs upon the successful completion of the CI workflow on the `main` branch.
* **Strategy:** Retrieves the exact Git commit SHA (`github.event.workflow_run.head_sha`) to pull the corresponding immutable image from GHCR. This prevents deployment mismatch issues caused by using the mutable `:latest` tag.

---

## 13. Local Kubernetes Deployment (Kind)

The application is deployed onto a local Kubernetes cluster named `cats-dogs-mlops` (context: `kind-cats-dogs-mlops`) on the control plane node `cats-dogs-mlops-control-plane`.

* **Manifests:** Configured using [`k8s/deployment.yaml`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/k8s/deployment.yaml) and [`k8s/service.yaml`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/k8s/service.yaml).
* **Networking:** The API container listens on port `8000`. The Kubernetes service exposes it locally on port `30080` using a `NodePort` service.
* **Policies & Constraints:** The deployment specifies CPU/memory requests and limits, readiness/liveness probes, and uses immutable image tags.

---

## 14. Local Docker Registry Integration

To enable image pulling without exposing our Kind cluster to public registry network limitations, we run a local Docker registry container named `kind-registry` on `localhost:5001`.
* **Internal Routing:** Within the Kind cluster network, this registry is accessible at `http://kind-registry:5000`.
* **Containerd Mapping:** Kind's containerd config maps references to `localhost:5001` directly to `http://kind-registry:5000`. This allows Kubernetes to resolve and pull local registry images seamlessly.

---

## 15. Image Transfer Flow: GHCR → Local Registry → Kubernetes

To bridge GitHub Container Registry (GHCR) and our local Kind cluster, the CD pipeline transfers the image through the following sequence:

```text
GHCR
 │
 │ 1. Pull exact Git SHA image
 ▼
Docker Runner Host
 │
 │ 2. Retag image as localhost:5001/cats-dogs-api:<FULL_SHA>
 ▼
Local Docker Registry (localhost:5001)
 │
 │ 3. Kubernetes / containerd pull from local endpoint
 ▼
Kind Kubernetes Cluster
 │
 ▼
cats-dogs-api Deployment Pods
```

By deploying the exact commit SHA tag (`localhost:5001/cats-dogs-api:<FULL_SHA>`) rather than `:latest`, we ensure deterministic deployments and eliminate version conflicts.

---

## 16. Kubernetes Health Checks

To maintain high availability, the deployment defines readiness and liveness probes targeting the FastAPI application:
* **Readiness Probe (`GET /health`):** Ensures Kubernetes only routes traffic to the pod once the model weights have successfully loaded into memory on startup.
* **Liveness Probe (`GET /health`):** Enables Kubernetes to detect and automatically restart unhealthy containers if the API stops responding.

---

## 17. Post-Deployment Smoke Testing

Once deployment rollout completes, a validation smoke test is run via the script [`scripts/smoke_test.py`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/scripts/smoke_test.py).
* **Validation Checks:** Verifies the health check endpoint, checks that the model is loaded, sends a sample prediction request, and validates that class probabilities and confidence scores are correct.
* **Failure Handling:** If any check fails, the script exits with a non-zero status code, causing the CD pipeline to fail.

To run the smoke test locally against a running API:
```bash
python scripts/smoke_test.py
```

---

## 18. Basic Monitoring and Logging

* **Server Logs:** The inference service logs request details (such as `GET /health` and `POST /predict`) to standard output. Review the logs from Kubernetes using:
  ```bash
  kubectl logs deployment/cats-dogs-api
  ```
* **Performance Metrics:** Access live request counters and latency snapshots at `GET /metrics`. This endpoint tracks total requests, success/failure counts, specific predictions (`cat` vs `dog`), and average latency.
* **Privacy:** To respect privacy, no uploaded images or raw input data are logged or stored.

---

## 19. Post-Deployment Model Performance Tracking

We track real-world model accuracy using the script [`scripts/post_deployment_evaluation.py`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/scripts/post_deployment_evaluation.py). 

* **How it works:** The script sends a labeled batch of 20 images (10 cats, 10 dogs) to the deployed API and computes evaluation metrics.
* **Metrics Recorded:** True vs. predicted labels, prediction confidence, latency per request, overall accuracy, per-class accuracy, and a confusion matrix.

---

## 20. Final Post-Deployment Results

Our post-deployment evaluation yielded the following performance metrics:

| Metric | Result |
| :--- | :--- |
| **Evaluation Images** | 20 |
| **Cat Images** | 10 |
| **Dog Images** | 10 |
| **Correct Predictions** | 16 |
| **Incorrect Predictions** | 4 |
| **Overall Accuracy** | 80% |
| **Cat Accuracy** | 70% |
| **Dog Accuracy** | 90% |
| **Average Confidence** | 72.16% |
| **Average Latency** | ~2089 ms |

### Confusion Matrix
```text
                         Predicted
                       Cat       Dog
Actual Cat              7         3
Actual Dog              1         9
```

The detailed metric payload is stored in the artifact [`artifacts/post_deployment_evaluation.json`](file:///e:/BITS%20Sem%203/MLOPS/Assignment%202/artifacts/post_deployment_evaluation.json).

---

## 21. Running the Application Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API Server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Verify Health:**
   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   ```
4. **Send a Test Prediction:**
   ```bash
   curl.exe -X POST "http://localhost:8000/predict" \
     -F "file=@tests/fixtures/cat.jpg"
   ```
5. **View Metrics:**
   ```powershell
   Invoke-RestMethod http://localhost:8000/metrics
   ```

---

## 22. Running Tests

To run the automated test suite locally:
```bash
pytest -v
```

To run the smoke test against a locally running API service (e.g., on port 8000):
```powershell
$env:SMOKE_TEST_URL="http://localhost:8000"
python scripts/smoke_test.py
```

---

## 23. Kubernetes Verification Commands

You can verify the state of your local Kubernetes deployment using the following commands:

* **Check cluster nodes:**
  ```bash
  kubectl get nodes
  ```
* **Verify deployment status:**
  ```bash
  kubectl get deployment cats-dogs-api
  ```
* **List running pods:**
  ```bash
  kubectl get pods -l app=cats-dogs-api -o wide
  ```
* **Describe service details:**
  ```bash
  kubectl get service cats-dogs-api
  ```
* **Inspect the deployed container image tag:**
  ```powershell
  kubectl get deployment cats-dogs-api \
    -o jsonpath="{.spec.template.spec.containers[0].image}"
  ```
* **Track rollout progress:**
  ```bash
  kubectl rollout status deployment/cats-dogs-api --timeout=600s
  ```

---

## 24. Accessing the Kubernetes API Locally

To interact with the API running inside the Kind cluster:

1. **Port-Forward the Service:**
   ```bash
   kubectl port-forward service/cats-dogs-api 8001:8000
   ```
2. **Test Endpoints (on port 8001):**
   * **Health Check:**
     ```powershell
     Invoke-RestMethod http://localhost:8001/health
     ```
   * **Prediction:**
     ```bash
     curl.exe -X POST "http://localhost:8001/predict" \
       -F "file=@tests/fixtures/cat.jpg"
     ```
   * **Metrics:**
     ```powershell
     Invoke-RestMethod http://localhost:8001/metrics
     ```

---

## 25. Reproducibility & Traceability

This pipeline uses immutable Git SHA image tags to map every deployed container directly to the code that produced it:

```text
Git commit
    │
    ▼
CI (GitHub Actions)
    │
    ├── Run tests
    ├── Validate model checksum
    └── Build Docker image
            │
            ▼
       GHCR: ghcr.io/varpeprasanna/bits-mtech-mlops-assignment-2:<FULL_SHA>
            │
            ▼
       CD (GitHub Actions)
            │
            ▼
       Local Registry: localhost:5001/cats-dogs-api:<FULL_SHA>
            │
            ▼
       Kind Kubernetes Pods
```

This strict linking provides full auditability, allowing teams to trace any running model in production back to the exact git commit, training run, and validation artifact.