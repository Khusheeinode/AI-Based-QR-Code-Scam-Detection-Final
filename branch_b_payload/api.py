import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from branch_b_payload.payload_pipeline import analyze_payload

app = FastAPI(
    title="QR Shield API",
    description="QR Scam Detection Backend",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PayloadRequest(BaseModel):
    payload: str


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "QR Shield API is running"
    }


@app.post("/analyze")
def analyze(request: PayloadRequest):

    try:
        payload = request.payload.strip()

        if not payload:
            raise HTTPException(
                status_code=400,
                detail="QR payload cannot be empty"
            )

        result = analyze_payload(
            payload,
            phishtank_api_key=os.getenv("PHISHTANK_API_KEY")
        )

        return {
            "success": True,
            "analysis": result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )