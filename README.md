# YOLOv8 Vehicle Speed Estimation 🚗💨

This project performs **vehicle detection, tracking, speed estimation, and counting** on video footage using **YOLOv8** and **OpenCV**.

A perspective transformation (bird-eye view) is applied to improve real-world speed estimation accuracy.

---

## 📌 Features

- Vehicle detection using YOLOv8
- Object tracking with ID assignment
- Vehicle counting inside a defined region
- Speed estimation (km/h)
- Perspective (bird-eye) transformation
- Real-time visualization

---

## 🧠 How It Works

1. Vehicles are detected using a pretrained YOLOv8 model.
2. Each detected object is assigned an ID based on spatial proximity.
3. A polygonal region defines the area of interest.
4. Speed is calculated only when vehicles enter this region.
5. Perspective transformation converts pixel movement into real-world distance.
6. Final speed is displayed in km/h.

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

---


