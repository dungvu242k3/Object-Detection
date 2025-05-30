# 🚗 Vehicle Detection API using YOLOv12 & FastAPI

Dự án cá nhân xây dựng hệ thống **phát hiện phương tiện giao thông** (ô tô, xe máy, xe tải, xe buýt, v.v.) từ hình ảnh đầu vào, sử dụng mô hình **YOLOv12** huấn luyện trên **dữ liệu tự thu thập**, triển khai qua REST API với **FastAPI**.

---

## 📌 Tính năng chính

- Phát hiện và phân loại các phương tiện giao thông trong ảnh.
- Triển khai mô hình YOLOv12 làm REST API.
- Nhận ảnh qua URL và trả về ảnh đã đánh dấu (bounding boxes).
- Xử lý và infer ảnh ngay trên CPU/GPU tùy thiết lập.

---

## 🧠 Công nghệ sử dụng

- [YOLOv12 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- PyTorch
- OpenCV
- FastAPI
- Uvicorn (ASGI server)
- Requests, NumPy

---

## 🗂️ Cấu trúc thư mục

```

vehicle-detect-api/
├── main.py              # FastAPI backend
├── best.pt              # Trained YOLOv12 model weights
├── requirements.txt     # All dependencies
├── README.md
└── ...

````

---

## ⚙️ Cài đặt môi trường

**1. Clone project**
```bash
git clone https://github.com/your-username/vehicle-detect-api.git
cd vehicle-detect-api
````

**2. Tạo môi trường ảo và cài dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏃‍♂️ Chạy API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📥 Sử dụng API

Gửi yêu cầu GET đến:

```
GET http://localhost:8000/detect/?image_url=https://example.com/image.jpg
```

**Trả về:** ảnh JPEG với bounding boxes và nhãn phương tiện.

---

## 🧪 Ví dụ sử dụng

```bash
curl -X GET "http://localhost:8000/detect/?image_url=https://upload.wikimedia.org/traffic.jpg" --output result.jpg
```

---

## 📁 Dữ liệu & Huấn luyện

* Dữ liệu ảnh được **tự thu thập** từ camera giao thông và video YouTube.
* Annotate bằng [Roboflow](https://roboflow.com) hoặc [LabelImg](https://github.com/tzutalin/labelImg).
* Chuẩn YOLOv12 (ảnh `.jpg`, nhãn `.txt`).
* Mô hình huấn luyện qua lệnh:

```bash
yolo detect train data=vehicle.yaml model=yolov12s.pt epochs=100 imgsz=640
