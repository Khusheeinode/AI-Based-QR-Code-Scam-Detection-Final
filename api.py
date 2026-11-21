import os
import base64
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from branch_a_image.gradcam import analyze_image
from branch_b_payload.payload_pipeline import analyze_payload
from fusion.fusion import fuse


app = FastAPI(
    title="QR Shield API",
    description="QR Code Scam Detection using CNN, Payload Analysis and Fusion",
    version="2.0"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "branch_a_image" / "qr_cnn_model.keras"

MAX_IMAGE_SIZE = 10 * 1024 * 1024

cnn_model = None


# --------------------------------------------------
# LOAD CNN MODEL
# --------------------------------------------------

def get_cnn_model():

    global cnn_model

    if cnn_model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"CNN model not found: {MODEL_PATH}"
            )

        cnn_model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )

    return cnn_model


# --------------------------------------------------
# EXTRACT AND RECTIFY QR IMAGE
# --------------------------------------------------

def extract_qr_image(image):

    detector = cv2.QRCodeDetector()

    detected, points = detector.detect(image)

    if not detected or points is None:
        raise ValueError(
            "QR code could not be detected in the uploaded image."
        )

    points = points.reshape(4, 2).astype(np.float32)

    output_size = 512

    destination = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    qr_image = cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size)
    )

    return qr_image


# --------------------------------------------------
# GRAD-CAM IMAGE ENCODING
# --------------------------------------------------

def encode_overlay(overlay):

    if overlay is None:
        return None

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    success, buffer = cv2.imencode(
        ".png",
        overlay_bgr
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "QR Shield API is running",
        "status": "active"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_exists": MODEL_PATH.exists()
    }


# --------------------------------------------------
# QR ANALYSIS
# --------------------------------------------------

@app.post("/analyze")
async def analyze(
    payload: str = Form(...),
    image: UploadFile = File(...)
):

    # ----------------------------------------------
    # VALIDATE PAYLOAD
    # ----------------------------------------------

    payload = payload.strip()

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="QR payload cannot be empty."
        )

    # ----------------------------------------------
    # VALIDATE IMAGE
    # ----------------------------------------------

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    # ----------------------------------------------
    # DECODE IMAGE
    # ----------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    original_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if original_image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image."
        )

    # ----------------------------------------------
    # EXTRACT QR REGION
    # ----------------------------------------------

    try:

        qr_image = extract_qr_image(
            original_image
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    # ----------------------------------------------
    # BRANCH A: CNN IMAGE ANALYSIS
    # ----------------------------------------------

    try:

        model = get_cnn_model()

        image_result = analyze_image(
            model,
            qr_image
        )

        image_risk_score = image_result["score"]

        gradcam_overlay = encode_overlay(
            image_result["overlay"]
        )

    except Exception as exc:

        print("CNN ERROR:", str(exc))

        raise HTTPException(
            status_code=500,
            detail="CNN image analysis failed."
        )

    # ----------------------------------------------
    # BRANCH B: PAYLOAD ANALYSIS
    # ----------------------------------------------

    try:

        payload_result = analyze_payload(
            payload,
            phishtank_api_key=os.getenv(
                "PHISHTANK_API_KEY"
            )
        )

    except Exception as exc:

        print("PAYLOAD ANALYSIS ERROR:", str(exc))

        raise HTTPException(
            status_code=500,
            detail="Payload analysis failed."
        )

    # ----------------------------------------------
    # FUSION: COMBINE BOTH BRANCHES
    # ----------------------------------------------

    try:

        final_result = fuse(
            image_risk_score,
            payload_result
        )

    except Exception as exc:

        print("FUSION ERROR:", str(exc))

        raise HTTPException(
            status_code=500,
            detail="Risk fusion failed."
        )

    # ----------------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------------

    return {

        "success": True,

        "analysis": final_result,

        "payload_analysis": payload_result,

        "gradcam": {

            "available": gradcam_overlay is not None,

            "image_base64": gradcam_overlay,

            "status": image_result["gradcam_ok"],

            "note": image_result["gradcam_note"]

        }

    }