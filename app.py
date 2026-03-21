import os
import cv2
import sys
import torch
import time
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request
from multiprocessing import Process, Manager, Queue, freeze_support
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import torch.backends.cudnn as cudnn


# ================= CONFIG =================

WEIGHTS = "crowdhuman_yolov5m.pt"

# RTSP CAMERA (Production)
#VIDEO_SOURCE = "rtsp://username:password@camera_ip:554/stream"

CAMERAS = {
    1: {
        "source": "video1.mp4",
        "entry_line": 500
    },
    2: {
        "source": "video2.mp4",
        "entry_line": 450
    },
    3: {
        "source": "video3.mp4",
        "entry_line": 450
    },
    4: {
        "source": "video4.mp4",
        "entry_line": 400
    },
}

IMG_SIZE = 384
CONF_THRES = 0.4
IOU_THRES = 0.3
ROI_MARGIN = 120
BATCH_SIZE = 25

cudnn.benchmark = True

app = Flask(__name__)

# ================= DATABASE =================

def init_db():
    conn = sqlite3.connect("events.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ================= INFERENCE PROCESS =================

def inference_worker(frame_queue, result_dict):

    device = select_device('0')
    model = DetectMultiBackend(WEIGHTS, device=device, fp16=True)
    stride = model.stride
    model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))

    while True:
        cam_id, roi, roi_top = frame_queue.get()

        img = letterbox(roi, IMG_SIZE, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)

        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

        detections = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], roi.shape).round()
                for *xyxy, conf, cls in det:
                    if int(cls) != 1:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    y1 += roi_top
                    y2 += roi_top
                    detections.append(([x1, y1, x2-x1, y2-y1], float(conf)))

        result_dict[cam_id] = detections

# ================= CAMERA PROCESS =================

def camera_worker(cam_id, config, frame_dict, frame_queue, result_dict):

    source = config["source"]
    entry_line = config["entry_line"]

    tracker = DeepSort(max_age=10, n_init=2, embedder="mobilenet", half=True, bgr=True)

    conn = sqlite3.connect("events.db", check_same_thread=False)
    cursor = conn.cursor()
    event_buffer = []

    previous_y = {}
    counted_ids = set()
    total_entries = 0

    os.makedirs(f"output/camera_{cam_id}", exist_ok=True)

    #cap = cv2.VideoCapture(source)                                    #For video
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)                   #---------------------------------------For Camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                               #------------------------------------For Camera
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    current_hour = None
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_id} finished.")
            break

        now = datetime.now().strftime("%Y-%m-%d_%H")
        if current_hour != now:
            if out:
                out.release()
            filename = f"output/camera_{cam_id}/{now}.mp4"
            out = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )
            current_hour = now

        roi_top = max(0, entry_line - ROI_MARGIN)
        roi_bottom = min(height, entry_line + ROI_MARGIN)
        roi = frame[roi_top:roi_bottom, :]

        frame_queue.put((cam_id, roi, roi_top))

        while cam_id not in result_dict:
            time.sleep(0.001)

        detections = result_dict.pop(cam_id)

        formatted_detections = [
            (d[0], d[1], "head") for d in detections
        ]

        tracks = tracker.update_tracks(
            formatted_detections,
            frame=frame
        )


        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            cy = int((t + b) / 2)

            if track.track_id not in previous_y:
                previous_y[track.track_id] = cy

            if previous_y[track.track_id] < entry_line and cy >= entry_line:
                if track.track_id not in counted_ids:
                    counted_ids.add(track.track_id)
                    total_entries += 1
                    event_buffer.append((cam_id,))

            previous_y[track.track_id] = cy

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0,255,0), 2)

        if len(event_buffer) >= BATCH_SIZE:
            cursor.executemany(
                "INSERT INTO entry_events (camera_id) VALUES (?)",
                event_buffer
            )
            conn.commit()
            event_buffer.clear()

        cv2.line(frame, (0, entry_line), (width, entry_line), (0,0,255), 3)
        cv2.putText(frame, f"TOTAL: {total_entries}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)

        out.write(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_dict[cam_id] = buffer.tobytes()

# ================= STREAM =================

def generate(cam_id):
    while True:
        if cam_id in app.config["frames"]:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   app.config["frames"][cam_id] + b'\r\n')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS.keys())

@app.route('/video/<int:camera_id>')
def video(camera_id):
    return Response(generate(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/report", methods=["GET", "POST"])
def report():
    conn = sqlite3.connect("events.db")
    cameras = list(CAMERAS.keys())
    results = None

    if request.method == "POST":
        selected = request.form.getlist("camera")
        start = request.form.get("start_time")
        end = request.form.get("end_time")

        query = "SELECT camera_id, COUNT(*) FROM entry_events WHERE 1=1"
        params = []

        if selected:
            query += " AND camera_id IN ({})".format(",".join(["?"]*len(selected)))
            params.extend(selected)

        if start:
            start = start.replace("T"," ") + ":00"
            query += " AND timestamp >= ?"
            params.append(start)

        if end:
            end = end.replace("T"," ") + ":59"
            query += " AND timestamp <= ?"
            params.append(end)

        query += " GROUP BY camera_id"
        results = conn.execute(query, params).fetchall()

    conn.close()

    return render_template("report.html",
                           cameras=cameras,
                           results=results)



# ================= MAIN =================

if __name__ == "__main__":
    freeze_support()
    init_db()

    manager = Manager()
    frames = manager.dict()
    results = manager.dict()
    frame_queue = Queue(maxsize=20)

    app.config["frames"] = frames

    infer_process = Process(
        target=inference_worker,
        args=(frame_queue, results)
    )
    infer_process.start()

    processes = []
    for cam_id, config in CAMERAS.items():
        p = Process(
            target=camera_worker,
            args=(cam_id, config, frames, frame_queue, results)
        )
        p.start()
        processes.append(p)

    app.run(host="0.0.0.0", port=5000, threaded=True)
