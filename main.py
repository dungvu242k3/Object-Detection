import os
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH).to("cpu")


def download_image(url: str):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {e}")

def run_inference(img):
    results = model(img)
    annotated_img = results[0].plot()
    return annotated_img

@app.get("/detect/")
def detect_api(image_url: str = Query(..., description="URL của ảnh")):
    img = download_image(image_url)
    annotated_img = run_inference(img)
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
