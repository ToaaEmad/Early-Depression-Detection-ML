# Early Depression Detection Using Computer Vision and NLP

This project explores an AI-based approach for early depression detection by combining two complementary sources of information:

- Computer Vision for analyzing facial expressions and emotional cues
- Natural Language Processing for analyzing text-based mental health signals

## Project Overview

Early depression detection can benefit from multiple modalities. This project investigates how machine learning and deep learning techniques can be applied to:

1. Detect emotional expressions from facial images
2. Classify mental-health-related text using NLP techniques
3. Provide a foundation for future deployment in assistive or clinical-support systems

## What's Included

- Computer Vision pipeline for facial expression recognition
- NLP pipeline for text classification and model comparison
- FastAPI-based inference service for the vision component
- Architecture diagrams and project reports

## Repository Structure

```text
Early-Depression-Detection-ML/
├── CV/                          # Computer vision trial/training notebooks and model assets
├── CV_New/                      # FastAPI deployment for the FER model
├── NLP/                         # NLP notebooks for preprocessing, EDA, ML, DL, transformers, and deployment
├── Diagrams/                    # System and workflow diagrams
├── Reports/                     # Project reports and documentation
└── README.md                    # Project overview
```

## Technologies Used

### Computer Vision
- Python
- PyTorch / TorchVision
- timm (EfficientNet-B4, pretrained on ImageNet, fine-tuned on AffectNet)
- facenet-pytorch (MTCNN for face detection + alignment)
- pytorch-grad-cam (explainability)
- OpenCV / PIL
- FastAPI + Uvicorn/Gunicorn
- ONNX (portable export of trained weights)

### Natural Language Processing
- Python
- Pandas and NumPy
- Scikit-learn
- Deep learning models
- Transformer-based models
- Jupyter notebooks

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Early-Depression-Detection-ML
```

### 2. Create a Python environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

For the computer vision API:

```bash
cd CV_New
pip install -r requirements.txt
```

For NLP experiments, open the notebooks in the NLP folder and install the required libraries as needed for each notebook.

## System Requirements

### Software
- Python 3.9 or newer
- pip and virtualenv
- Jupyter Notebook or JupyterLab
- Git and Git LFS (required — the trained model weights exceed GitHub's normal file size limits)

### Hardware
- CPU: modern multi-core processor is sufficient for basic experimentation and for running inference in production
- GPU: optional, recommended for faster training only — inference runs fine on CPU (~0.5–2s per image)
- RAM: 8 GB minimum locally; 2 GB minimum on the hosting plan (see Azure section below)

## Configuration Instructions

- Ensure the model assets and label files are present in the relevant folders before running inference.
- If you are using a different environment, update paths if needed when launching the API or notebooks.
- For the FastAPI app, the required files (`fer_model.pth`, `label_map.json`) are expected to be present in the same directory as `app.py`.

## Model Training Pipeline (Computer Vision)

The FER model was trained on Kaggle using a GPU-accelerated notebook (`CV/AffectNet_FER_Kaggle.ipynb`). Summary of the pipeline:

| Stage | Choice | Why |
|---|---|---|
| Backbone | EfficientNet-B4 (`tf_efficientnet_b4_ns`), ImageNet-pretrained | Strong accuracy-to-compute ratio; fits Kaggle's free GPU quota while still fine-tuning well on a specialized task |
| Face detection/alignment | MTCNN (facenet-pytorch) | Crops and aligns the face before classification so background clutter cannot influence the prediction |
| Classes | Sadness, Neutral, Fear (proxy for "Anxiety" — AffectNet has no native Anxiety label), Disgust | Scoped to the subset most relevant to the project's mental-health-adjacent focus |
| Augmentation | RandAugment, horizontal flip, random erasing | Improves robustness to real-world lighting/pose variation |
| Loss function | Focal Loss (class-weighted) | AffectNet is imbalanced across emotion classes; Focal Loss down-weights easy examples and focuses learning on minority classes |
| Optimizer / scheduler | AdamW + Cosine Annealing | Stable convergence with proper weight decay handling |
| Explainability | Grad-CAM (pytorch-grad-cam) | Visualizes which facial regions (eyes, brow, mouth, nose) drove each prediction |
| Export | `.pth` (PyTorch) and `.onnx` | `.pth` used in the deployed API; `.onnx` provided for portability to other runtimes |

**Known limitation:** the model was not trained on "Happy" or "Surprise" classes. A genuinely happy expression will still be forced into one of the four trained categories, sometimes with high (overconfident) probability. A confidence threshold (see below) mitigates this by returning "Uncertain" instead of a misleading label.

## Execution Guide

### Run the Computer Vision API locally

From the `CV_New` folder:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Test `/predict` by expanding it in Swagger UI, clicking **Try it out**, uploading an image, then **Execute**.

### Run the NLP workflow locally

Open the notebooks in the NLP folder in order:

1. Data_Preprocessing.ipynb
2. EDA.ipynb
3. Machine Learning.ipynb
4. Deep Learning.ipynb
5. Transformer Model.ipynb
6. Model Comparison and Selection.ipynb
7. FastAPI Deployment.ipynb

## API Documentation

The computer vision service provides interactive API documentation through Swagger UI.

- Swagger UI: `/docs`
- Health check: `/health`
- Inference: `POST /predict` — accepts `multipart/form-data` with a `file` field (an image); returns the predicted expression, a display-friendly name, a confidence score, the full probability distribution across all classes, and a disclaimer that the result is an expression classification, not a clinical diagnosis.

If the model's top prediction confidence is below a set threshold (default 0.60), the API returns `"label": "uncertain"` instead of forcing a possibly-wrong label — this specifically helps with out-of-scope expressions (e.g. happiness) that don't map cleanly onto the four trained classes.

## Deployment — Azure App Service (CV component)

The FastAPI inference service is deployed to **Azure App Service (Linux, Web App)**, connected to this GitHub repository through Azure's built-in **Deployment Center**, so every push to the configured branch triggers an automatic redeploy.

### 1. Repository layout for deployment

The following files live together in the `CV_New/` deployment folder so Azure's build step can find everything it needs:

```text
CV_New/
├── app.py              # FastAPI app (see API Documentation above)
├── requirements.txt    # Pinned dependencies, CPU-only torch build
├── startup.sh           # Azure App Service startup command reference
├── .gitattributes       # Git LFS tracking rules for large model files
├── fer_model.pth        # Trained model weights (tracked via Git LFS)
└── label_map.json       # Class index → display name mapping
```

### 2. Git LFS (required for the model weights)

`fer_model.pth` is tens of MB, which sits close to/over GitHub's plain file-size comfort zone, so it's tracked with Git LFS rather than committed directly:

```bash
git lfs install
git add .gitattributes
git add fer_model.pth label_map.json app.py requirements.txt startup.sh
git commit -m "Add FER model deployment files"
git push
```

### 3. Azure Web App configuration

In the Azure Portal, on the Web App already connected to this repo via Deployment Center:

- **Runtime stack:** Python 3.9+ (Linux)
- **Startup Command** (`Configuration` → `General settings`):
  ```
  gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT app:app --timeout 120
  ```
  A single worker (`-w 1`) is used deliberately — the loaded model consumes a meaningful chunk of RAM, and multiple workers each load their own copy of the model.
- **App Service Plan:** at least **B2** (2 GB RAM). Smaller plans (F1/B1) may fail to load the model and return an Application Error.

### 4. Verifying the deployment

After a push completes (check `Deployment Center` → latest deployment shows **Success**):

```
GET https://<MindGaurd_AI>.azurewebsites.net/health   -> {"status": "ok"}
GET https://<MindGaurd_AI>.azurewebsites.net/docs      -> interactive Swagger UI
```

If `/health` doesn't respond, check `Monitoring` → `Log stream` in the Azure Portal for the startup error (most commonly a `requirements.txt` resolution issue or insufficient RAM on the plan).

### 5. Frontend integration (Lovable)

The deployed endpoint consumed by the frontend is:

```
POST https://<MindGaurd_AI>.azurewebsites.net/predict
Content-Type: multipart/form-data
Field: file  (the image)
```

CORS is currently open (`allow_origins=["*"]`) for development; before final submission this should be restricted to the actual Lovable app domain in `app.py`'s `CORSMiddleware` configuration.

## Notes

- The system is designed to support research and experimentation.
- Outputs should be interpreted cautiously and **not used as a stand-alone medical assessment**. The CV component classifies facial *expressions* (Sadness, Neutral, Fear/Anxiety-proxy, Disgust) — it does not diagnose depression, anxiety, or any clinical condition.
- Further validation with real-world datasets and domain experts is recommended before any deployment in sensitive environments.

## License

This project is provided for educational and research purposes.
