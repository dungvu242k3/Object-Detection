import io
import os
from io import BytesIO

import cv2
import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

base_dir = "C:/Users/dungv/Object-Detection"
class_names = ["Ambulance", "Bus", "Car", "Motorbike", "Truck"]

yaml_content = {
    "path": base_dir,
    "train": "train/images",
    "val": "test/images",
    "names": class_names,
    "nc": len(class_names)
}

yaml_path = os.path.join(base_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)

model = YOLO("yolo12n.pt")

app = FastAPI()
label_map = {0: 'Ambulance', 1: 'Bus', 2: 'Car', 3: 'Motorbike', 4: 'Truck'}

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
    model_train = YOLO("yolo12n.pt")
    model_train.train(data=yaml_path, epochs=100, imgsz=640, batch=32)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
