import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights).to(device)
model.eval()

image_path = "C:/Users/dungv/Pictures/Glossy_Horse_1920x.webp"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)
image_tensor = image_tensor.to(device)
with torch.no_grad():
    outputs = model(image_tensor)

boxes = outputs[0]["boxes"]
labels = outputs[0]["labels"]
scores = outputs[0]["scores"]

threshold = 0.7  
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

names = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

for box, label, score in zip(boxes, labels, scores):
    if score > threshold:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        label_name = names[label.item()]
        ax.text(x1, y1, f"{label_name}: {score:.2f}", color='white',
                bbox=dict(facecolor='red', alpha=0.5))

plt.axis("off")
plt.show()
