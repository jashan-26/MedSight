"""
FastAPI Server for PACS AI Radiology Decision-Support Web Application.
Navy Blue & Cream Clinical Frontend & REST API Endpoints.
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    DISEASE_CLASSES, AVAILABLE_MODELS, RISK_LEVELS,
    MEDICAL_DISCLAIMER, ANATOMICAL_MAP
)
from processing.preprocessor import ImagePreprocessor
from models.model_manager import ModelManager
from reporting.pdf_generator import generate_pdf_report
from utils.sample_generator import get_sample_xrays
from utils.metrics_store import MODEL_PERFORMANCE_METRICS, DATASET_DISTRIBUTION, CONFUSION_MATRIX_EFFICIENTNET

app = FastAPI(
    title="Medical PACS AI Radiology Screening API",
    description="REST API for Chest X-Ray Multi-Label Triage & Grad-CAM Explainability",
    version="2.5.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Singleton Model Manager & Preprocessor
model_mgr = ModelManager()
preprocessor = ImagePreprocessor()
sample_xrays = get_sample_xrays()


def numpy_to_base64(img_np):
    """Converts RGB numpy array to base64 PNG data URL string."""
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Medical PACS AI Web Portal</h1><p>Static index.html not found.</p>"


@app.get("/api/samples")
async def list_samples():
    """Returns dictionary of available preloaded sample radiograph names."""
    return {"samples": list(sample_xrays.keys())}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(None),
    sample_name: str = Form("None"),
    model_name: str = Form("EfficientNet-B0"),
    auto_crop: bool = Form(True),
    enable_clahe: bool = Form(False),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    colormap: str = Form("JET"),
    alpha: float = Form(0.5)
):
    """
    Inference endpoint: Preprocesses radiograph, computes PyTorch predictions and Grad-CAM maps.
    """
    target_img = None
    source_name = "Uploaded_Scan.png"

    if file and file.filename:
        contents = await file.read()
        target_img = Image.open(io.BytesIO(contents)).convert("RGB")
        source_name = file.filename
    elif sample_name in sample_xrays:
        sample_path = sample_xrays[sample_name]
        target_img = Image.open(sample_path).convert("RGB")
        source_name = os.path.basename(sample_path)
    else:
        raise HTTPException(status_code=400, detail="Please provide either an uploaded file or select a valid sample.")

    # 1. Preprocess
    _, proc_np, input_tensor = preprocessor.preprocess(
        target_img,
        auto_crop=auto_crop,
        enable_clahe=enable_clahe,
        brightness=brightness,
        contrast=contrast
    )

    # 2. Model Inference & Grad-CAM
    results = model_mgr.predict(
        input_tensor,
        model_name=model_name,
        original_img_np=proc_np,
        colormap_choice=colormap,
        alpha_blend=alpha
    )

    # 3. Convert images to Base64
    proc_b64 = numpy_to_base64(proc_np)
    heatmap_b64 = numpy_to_base64(results["colored_heatmap"])
    overlay_b64 = numpy_to_base64(results["overlay_image"])

    return JSONResponse({
        "source_name": source_name,
        "model_name": model_name,
        "primary_finding": results["primary_finding"],
        "confidence_pct": results["confidence_pct"],
        "risk_level": results["risk_level"],
        "probabilities": results["probabilities"],
        "uncertainty": results["uncertainty"],
        "anatomical_attention": results["anatomical_attention"],
        "images": {
            "preprocessed": proc_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64
        }
    })


@app.post("/api/report")
async def generate_report_pdf(
    sample_name: str = Form("None"),
    model_name: str = Form("EfficientNet-B0"),
    file: UploadFile = File(None)
):
    """Generates printable PDF screening report and streams buffer."""
    target_img = None
    source_name = "Uploaded_Scan.png"

    if file and file.filename:
        contents = await file.read()
        target_img = Image.open(io.BytesIO(contents)).convert("RGB")
        source_name = file.filename
    elif sample_name in sample_xrays:
        sample_path = sample_xrays[sample_name]
        target_img = Image.open(sample_path).convert("RGB")
        source_name = os.path.basename(sample_path)
    else:
        sample_path = list(sample_xrays.values())[0]
        target_img = Image.open(sample_path).convert("RGB")
        source_name = os.path.basename(sample_path)

    _, proc_np, input_tensor = preprocessor.preprocess(target_img)
    results = model_mgr.predict(input_tensor, model_name=model_name, original_img_np=proc_np)

    pdf_bytes = generate_pdf_report(
        analysis_results=results,
        original_img_np=proc_np,
        overlay_img_np=results["overlay_image"],
        patient_id="PATIENT-92811",
        scan_name=source_name
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=PACS_Report_{results['primary_finding']}.pdf"}
    )


if __name__ == "__main__":
    print("Starting Medical PACS AI Web Server on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
