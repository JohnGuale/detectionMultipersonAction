import os, cv2, json, base64
from flask import Flask, request, Response, jsonify, send_file
from collections import defaultdict, deque
import numpy as np
import torch
from ultralytics import YOLO
import joblib

# -----------------------------
# Configuración
# -----------------------------
app = Flask(__name__)
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Modelos cargados (ejemplo)
pose_model = YOLO("yolov8s-pose.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
lstm_model = torch.load("models/lstm_model.pth", map_location=device)
lstm_model.eval()

# Memoria temporal de detecciones
detecciones_global = []

# -----------------------------
# Funciones auxiliares
# -----------------------------
def run_pose_and_lstm(frame, dimension="2D"):
    """
    Procesa un frame con YOLO + LSTM y retorna lista de detecciones.
    """
    res = pose_model.predict(frame, device=device, verbose=False)
    detections = []
    if res and res[0].keypoints is not None:
        kps_xy = res[0].keypoints.xy.cpu().numpy()
        # Aquí simplificamos: si hay keypoints, asumimos detección
        for kp in kps_xy:
            # Ejemplo: si la mano está oculta, etc.
            # En tu pipeline real usarías rf_features_from_window + lstm_predict_with_conf
            detections.append("excessive_gaze")  # placeholder
    return detections

# -----------------------------
# Endpoints
# -----------------------------
@app.route('/save-video', methods=['POST'])
def save_video():
    file = request.files['video']
    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    return jsonify({"filename": filename})

@app.route('/stream_frames/<filename>/<int:skip>/<dimension>')
def stream_frames(filename, skip, dimension):
    path = os.path.join(UPLOAD_DIR, filename)
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def generate():
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                yield "data: EOF\n\n"
                break

            frame_idx += 1
            if skip > 0 and frame_idx % skip != 0:
                continue

            detections = run_pose_and_lstm(frame, dimension)
            if detections:
                detecciones_global.append({
                    "timestamp": frame_idx / cap.get(cv2.CAP_PROP_FPS),
                    "behaviors": detections
                })

            # convertir frame a JPEG base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "progress": int((frame_idx / total_frames) * 100),
                "frame": frame_b64,
                "detections": detections
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/detecciones')
def get_detecciones():
    return jsonify(detecciones_global)

@app.route('/processed-video/<filename>')
def get_processed_video(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return "Video no encontrado", 404
    return send_file(path, mimetype='video/webm')

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
