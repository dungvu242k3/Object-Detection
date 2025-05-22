import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov5s.pt").to(device)
model.eval()

image_path = "C:/Users/dungv/Pictures/Glossy_Horse_1920x.webp"
image = Image.open(image_path).convert("RGB")
results = model(image_path)[0]

boxes = results.boxes.xyxy.cpu()
scores = results.boxes.conf.cpu()
labels = results.boxes.cls.cpu().int()

threshold = 0.7
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

names = model.names

for box, label, score in zip(boxes, labels, scores):
    if score > threshold:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        label_name = names[label.item()]
        ax.text(x1, y1, f"{label_name}: {score:.2f}", color='white',bbox=dict(facecolor='red', alpha=0.5))

plt.axis("off")
plt.tight_layout()
plt.show()
