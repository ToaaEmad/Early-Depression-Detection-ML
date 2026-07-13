# Early Depression Detection Using Computer Vision and NLP

This project explores an AI-based approach for early depression detection by combining two complementary sources of information:

- Computer Vision for analyzing facial expressions and emotional cues
- Natural Language Processing for analyzing text-based mental health signals

## Project Overview

Early depression detection can benefit from multiple modalities. This project investigates how machine learning and deep learning techniques can be applied to:

1. Detect emotional expressions from facial images
2. Classify mental-health-related text using NLP techniques
3. Provide a foundation for future deployment in assistive or clinical-support systems

## What’s Included

- Computer Vision pipeline for facial expression recognition
- NLP pipeline for text classification and model comparison
- FastAPI-based inference service for the vision component
- Architecture diagrams and project reports

## Repository Structure

```text
Early-Depression-Detection-ML/
├── CV/                          # Computer vision Trail training notebooks and model assets
├── CV_New/                      # FastAPI deployment for the FER model
├── NLP/                         # NLP notebooks for preprocessing, EDA, ML, DL, transformers, and deployment
├── Diagrams/                    # System and workflow diagrams
├── Reports/                     # Project reports and documentation
└── README.md                    # Project overview
```

## Technologies Used

### Computer Vision
- Python
- TensorFlow / Keras
- OpenCV / PIL
- FastAPI
- PyTorch / TorchVision
- ONNX / model inference assets

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
venv\\Scripts\\activate
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
- Git

### Hardware
- CPU: modern multi-core processor is sufficient for basic experimentation
- GPU: optional but recommended for faster model training and inference
- RAM: 8 GB minimum, 16 GB recommended

## Installation Steps

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd Early-Depression-Detection-ML
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   venv\\Scripts\\activate
   ```

3. Install dependencies for the computer vision API
   ```bash
   cd CV_New
   pip install -r requirements.txt
   ```

4. Install notebook dependencies if you plan to run the NLP workflow
   ```bash
   pip install jupyter notebook
   ```

## Configuration Instructions

- Ensure the model assets and label files are present in the relevant folders before running inference.
- If you are using a different environment, update paths if needed when launching the API or notebooks.
- For the FastAPI app, the required files are expected to be present in the same directory as the application entry point.

## Execution Guide

### Run the Computer Vision API locally

From the CV_New folder:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

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

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Executable Files & Deployment Link

### Compiled / Packaged Application
- The project is deployed as a cloud-hosted application on Azure rather than as a standalone executable file.
- The backend service is exposed through Azure and connected to a React-based frontend website.

### Deployed Web / Mobile Application
- A deployed web application is available through Azure.
- The system is integrated with a React website for user interaction and access to the deployed service.
- If you have a public Azure URL, it can be added here for direct access.

## Running the Computer Vision API

From the CV_New folder:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000/docs for the FastAPI Swagger UI
- http://localhost:8000/health for a basic health check

## Running the NLP Workflow

Open the notebooks in the NLP folder in order:

1. Data_Preprocessing.ipynb
2. EDA.ipynb
3. Machine Learning.ipynb
4. Deep Learning.ipynb
5. Transformer Model.ipynb
6. Model Comparison and Selection.ipynb
7. FastAPI Deployment.ipynb

## Notes

- The system is designed to support research and experimentation.
- Outputs should be interpreted cautiously and not used as a stand-alone medical assessment.
- Further validation with real-world datasets and domain experts is recommended before any deployment in sensitive environments.

## License

This project is provided for educational and research purposes.
