# Chest X-Ray Analysis for Disease Detection and Heatmap Generation

This repository contains code and resources for analyzing chest X-ray images to detect diseases and generate heatmaps highlighting areas of interest. The project leverages deep learning techniques to classify chest X-rays and visualize the results.

**⚠️ Medical Disclaimer**:
**This application is for research and educational purposes only. It is not intended for clinical use, medical diagnosis, or treatment decisions. Always consult qualified healthcare professionals for medical advice.**

Demo Video : https://www.loom.com/share/0f9b12e5c1234c1eae47f7535b1e5d71?sid=bc8a49c7-469b-4b2e-8e31-b6d4b4445e30

- dataset :  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

## Run using docker-compose

```
docker compose up -d
```
you will be able to access the frontend at http://localhost:3000 and backend at http://localhost:8007/

# Frontend 

- The frontend is built using React and provides a user-friendly interface for uploading chest X-ray images and viewing the analysis results.

Running the Frontend:
1. Navigate to the frontend directory:
2. Install the required dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm run dev
   ```
4. Open your web browser and go to `http://localhost:5173` to access the application.

# Backend
# Alphagrid Medical AI Backend

A FastAPI-based backend service for chest X-ray analysis using deep learning. This application provides automated disease prediction and visual explanations through Grad-CAM heatmaps.

## Features

- **Disease Prediction**: Top-5 disease probabilities using pretrained DenseNet-121
- **Visual Explanations**: Grad-CAM heatmap overlays showing regions of interest
- **RESTful API**: Clean FastAPI endpoints with JSON responses
- **File Upload**: Secure handling of chest X-ray images (PNG/JPG)

## Model Information

This backend uses the **TorchXRayVision DenseNet-121** model with the following specifications:
- **Model**: `densenet121-res224-all`
- **Input**: Grayscale images resized to 224×224 pixels
- **Output**: Probabilities for 18 chest X-ray findings
- **Training Data**: NIH ChestX-ray14, CheXpert, MIMIC-CXR datasets

### Supported Findings
The model can detect 18 different pathological findings including:
- Atelectasis
- Cardiomegaly  
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia
- No Finding
- And others

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   FastAPI        │    │   ML Models     │
│   (Frontend)    │────│   Backend        │────│   TorchXRayVision│
│                 │    │   Router         │    │   + Grad-CAM    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Grad-CAM Workflow
```
Input Image (Chest X-ray)
          │
          ▼
   ┌─────────────┐
   │ Convolution │  ← Early conv layers: detect edges, textures
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  Conv Layers│  ← Intermediate layers: detect patterns, shapes
   └─────────────┘
          │
          ▼
   ┌─────────────────────────┐
   │ Last Conv Layer (target)│  ← Captures high-level features
   └─────────────────────────┘
          │
          ▼
Gradients w.r.t target class
          │
          ▼
  Weighted feature maps
          │
          ▼
   ┌───────────────┐
   │  Heatmap Overlay│  ← Shows which regions contributed most
   └───────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- uvicorn (for dependency management)
### Environment Setup

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
# Using uv (recommended - faster)
uv pip sync requirements.txt

# Or using pip
pip install -r requirements.txt
```

3.  Run tests: `pytest tests/`

4. **Running the server:**
```
uvicorn app:app --host 0.0.0.0 --port 8007 --timeout-keep-alive 1800 --reload
```

### Configuration Options
- `--host 0.0.0.0`: Accept connections from any IP
- `--port 8007`: Server port
- `--timeout-keep-alive 1800`: Keep connections alive for 30 minutes
- `--reload`: Auto-reload on code changes (development only)
- `--workers 1`: Single worker (recommended for GPU models)

## API Documentation

### Base URL
```
http://localhost:8007/api
```

### Endpoints

#### 1. Predict Disease (`POST /api/predict`)

Analyze chest X-ray and return top-5 disease predictions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (PNG/JPG image)

**Response:**
```json
{
  "status": "success",
  "data": {
    "filename": "chest_xray.jpg",
    "predictions": [
      {
        "label": "Pneumonia",
        "probability": 0.847
      },
      {
        "label": "Infiltration", 
        "probability": 0.234
      }
    ],
    "top_prediction": {
      "label": "Pneumonia",
      "probability": 0.847
    }
  }
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8007/api/predict" \
  -H "accept: application/json" \
  -F "file=@chest_xray.jpg"
```

#### 2. Explain Prediction (`POST /api/explain`)

Generate Grad-CAM heatmap overlay for visual explanation.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (PNG/JPG image)

**Response:**
```json
{
  "status": "success",
  "data": {
    "filename": "chest_xray.jpg",
    "heatmap_image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "explained_prediction": {
      "label": "Pneumonia",
      "probability": 0.847,
      "class_index": 9
    },
    "image_info": {
      "original_size": [512, 512],
      "model_input_size": [224, 224]
    }
  }
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8007/api/explain" \
  -H "accept: application/json" \
  -F "file=@chest_xray.jpg"
```

## Code Structure

```
backend/
├── app.py                 # FastAPI application entry point
├── routers/
│   └── chest_xray.py     # API routes (provided code)
├── models/
│   ├── inference.py      # Prediction logic
│   ├── explain.py        # Grad-CAM implementation
│   └── xray_model.py     # Model loading utilities
├── utils/
│   ├── file_manager.py   # File upload/cleanup
│   ├── image_utils.py    # Image processing
│   └── response_utils.py # JSON response helpers
├── tests/
│   ├── test_api.py       # API endpoint tests
│   └── test_model.py     # Model loading tests
└── requirements.txt      # Dependencies
```

## Key Components

### Model Initialization
The application initializes the model once at startup:


Global variables maintain model state:
- `model`: DenseNet-121 model instance
- `labels`: List of 18 pathology labels
- `preprocess`: Image preprocessing pipeline
- `target_layer`: Last convolutional layer for Grad-CAM
- `device`: CUDA device if available, else CPU

### File Handling
- Secure file upload with validation
- Automatic cleanup after processing
- Support for PNG and JPG formats
- Memory-efficient processing

### Error Handling
- Structured JSON error responses
- Comprehensive logging
- Graceful fallbacks for missing resources
- HTTP status code compliance


## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Test model loading
python -m pytest tests/test_model.py -v

# Test API endpoints
python -m pytest tests/test_api.py -v
```

### Manual Testing
```bash
# Health check
curl -X GET "http://localhost:8007/api/health"

# Test prediction with sample image
curl -X POST "http://localhost:8007/api/predict" \
  -F "file=@sample_xray.jpg"
```

#### File Upload Errors
```
HTTP error in predict: File format not supported
```
**Solution**: Ensure file is PNG or JPG format

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit pull request

## Acknowledgments

- **TorchXRayVision**: Pretrained chest X-ray models
- **pytorch-grad-cam**: Grad-CAM implementation
- **FastAPI**: Modern web framework for APIs
- **Medical Datasets**: NIH ChestX-ray14, CheXpert, MIMIC-CXR


---

**Remember**: This is a research tool only. Never use for actual medical diagnosis or treatment decisions.
