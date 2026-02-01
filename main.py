from ultralytics import YOLO
import cv2
import numpy as np
import math

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

# Video dosyasını aç
cap = cv2.VideoCapture("video_path")

# FPS ve boyut bilgilerini al
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS geçerli değilse manuel düzelt
if fps <= 0 or fps > 120:
    print("⚠️ FPS değeri hatalı okundu, varsayılan 30 olarak ayarlanıyor.")
    fps = 30.0

print(f"Video boyutu: {width}x{height}")
print(f"FPS: {fps}")

# Görüntüleme boyutu
display_width = 600
display_height = 400

# --- Perspektif bölge noktaları ---
#  Selected points: [(194, 159), (479, 149), (565, 261), (140, 241)]
tl = (170, 139)
bl = (100, 241)
tr = (479, 149)
br =  (565, 261)
region_pts = np.array([tl, bl, br, tr], np.int32).reshape((-1, 1, 2))

# --- Perspektif dönüşümü hedef boyutu ---
bird_width, bird_height = 600, 400

src_pts = np.float32([tl, bl, br, tr])
dst_pts = np.float32([
    [0, 0],
    [0, bird_height],
    [bird_width, bird_height],
    [bird_width, 0]
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)


# 7.5 m genişlik referansı
real_width_m = 2.0
pixel_width = math.dist(bl, tl)
px_to_m = real_width_m / pixel_width

# Takip ve sayaçlar
track_history = {}
object_positions = {}
entered_ids = set()
total_entered = 0
object_id_counter = 0

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1

    frame = cv2.resize(frame, (600, 400))

    # YOLO tespiti (araç sınıfları)
    results = model(frame, classes=[2, 3, 5, 7])
    detections = results[0].boxes.data.cpu().numpy()

    annotated_frame = frame.copy()

    # Bölgeyi çiz
    cv2.polylines(annotated_frame, [region_pts], True, (0, 255, 0), 2)
    overlay = annotated_frame.copy()
    cv2.fillPoly(overlay, [region_pts], color=(0, 255, 0))
    annotated_frame = cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0)

    new_positions = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        center = (cx, cy)

        inside = cv2.pointPolygonTest(region_pts, center, False)
        matched_id = None

        # ID eşleştirme
        for oid, (px, py) in object_positions.items():
            if math.dist((px, py), (cx, cy)) < 60:
                matched_id = oid
                break

        if matched_id is None:
            object_id_counter += 1
            matched_id = object_id_counter

        new_positions[matched_id] = (cx, cy)

        # Alan giriş kontrolü
        if inside > 0 and matched_id not in entered_ids:
            total_entered += 1
            entered_ids.add(matched_id)

        # Sadece alan içindeyse hız hesapla
        if inside > 0:
            if matched_id not in track_history:
                track_history[matched_id] = []
            track_history[matched_id].append((cx, cy, frame_index))

            # Hız hesaplama (PERSPEKTİF DÜZELTMELİ)
            speed_text = ""
            if len(track_history[matched_id]) >= 2:
                (x1p, y1p, f1), (x2p, y2p, f2) = track_history[matched_id][-2], track_history[matched_id][-1]
                time_diff = (f2 - f1) / fps

                if time_diff > 0:
                    # --- Kuş bakışı koordinatları ---
                    pts_src = np.array([[[x1p, y1p]], [[x2p, y2p]]], dtype=np.float32)
                    pts_bird = cv2.perspectiveTransform(pts_src, M)
                    (bx1, by1), (bx2, by2) = pts_bird[0][0], pts_bird[1][0]

                    # --- Dönüştürülmüş piksel mesafesi ---
                    pixel_dist = math.dist((bx1, by1), (bx2, by2))

                    # --- Gerçek hıza dönüştür ---
                    speed_m_s = (pixel_dist * px_to_m) / time_diff
                    speed_kmh = speed_m_s * 3.6
                    speed_text = f"{speed_kmh:.1f} km/h"
                else:
                    speed_text = "..."
            else:
                speed_text = "..."

            # Görsel
            color = (0, 255, 255)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, f"ID {matched_id}", (int(x1), int(y1) - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, speed_text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    object_positions = new_positions

    # Sayaç
    cv2.putText(annotated_frame, f"Total Vehicles Entered: {total_entered}",
                (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Perspektif görünüm (bird-eye)
    bird_eye = cv2.warpPerspective(frame, M, (bird_width, bird_height))
    cv2.putText(bird_eye, "Perspective View", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüleri ayrı pencerelerde göster
    resized_main = cv2.resize(annotated_frame, (display_width, display_height))
    resized_bird = cv2.resize(bird_eye, (display_width, display_height))

    cv2.imshow("YOLO Speed View", resized_main)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()