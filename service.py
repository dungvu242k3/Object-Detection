import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.abspath("dataset")

train_images_dir = os.path.join(base_dir, "train", "images")
train_labels_dir = os.path.join(base_dir, "train", "labels")
val_images_dir = os.path.join(base_dir, "test", "images")
val_labels_dir = os.path.join(base_dir, "test", "labels")
yaml_path = os.path.join(base_dir, "data.yaml")
weights_path = os.path.join(base_dir, "best.pt")

class_names = ["Ambulance", "Bus", "Car", "Motorbike", "Truck"]

yaml_content = {
    "path": base_dir,
    "train": "train/images",
    "val": "test/images",
    "names": class_names,
    "nc": len(class_names)
}

os.makedirs(base_dir, exist_ok=True)
for sub in ["train/images", "train/labels", "test/images", "test/labels"]:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

if not Path(weights_path).exists():
    model = YOLO("yolov8n.pt")
    model.train(data=yaml_path, epochs=10, imgsz=640, batch=16, device=0)
    best_model = Path("runs/detect/train/weights/best.pt")
    if best_model.exists():
        best_model.rename(weights_path)

model = YOLO(weights_path)
app = FastAPI()

def download_image(url: str):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

@app.get("/predict/")
def predict(image_url: str = Query(...)):
    img = download_image(image_url)
    results = model.predict(source=img, save=False, imgsz=640, conf=0.25)
    annotated = results[0].plot()
    _, img_encoded = cv2.imencode(".jpg", annotated)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
