from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import onnxruntime as ort
import numpy as np
import cv2
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model once when the server starts
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

CLASS_NAMES = ["Car", "Cyclist", "Misc", "Pedestrian", "Person_sitting", "Tram", "Truck", "Van"]
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5


def preprocess(frame):
    """Resize frame to 640x640 and convert to the numeric format ONNX expects."""
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]  # BGR (OpenCV default) -> RGB (model's expected format)
    img = img.transpose(2, 0, 1)  # HWC -> CHW (channels-first, what the model expects)
    img = img.astype(np.float32) / 255.0  # scale pixel values 0-255 down to 0-1
    img = np.expand_dims(img, axis=0)  # add a "batch" dimension: shape becomes (1, 3, 640, 640)
    return img


def postprocess(output, frame_width, frame_height):
    """Turn the model's raw numeric output into actual (box, class, confidence) detections."""
    predictions = np.squeeze(output[0]).T  # reshape to (num_boxes, 4 + num_classes)

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

        # Scale box coordinates from 640x640 model-space back to the real frame size
        x1 = (cx - w / 2) / INPUT_SIZE * frame_width
        y1 = (cy - h / 2) / INPUT_SIZE * frame_height
        box_w = w / INPUT_SIZE * frame_width
        box_h = h / INPUT_SIZE * frame_height

        boxes.append([x1, y1, box_w, box_h])
        scores.append(float(confidence))
        class_ids.append(class_id)

    # Remove overlapping duplicate boxes for the same object (Non-Max Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    for i in indices:
        i = i if isinstance(i, (int, np.integer)) else i[0]
        results.append((boxes[i], scores[i], class_ids[i]))
    return results


def draw_boxes(frame, detections):
    for box, score, class_id in detections:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = f"/tmp/{uuid.uuid4()}_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        output = session.run(None, {input_name: input_tensor})
        detections = postprocess(output, width, height)
        annotated_frame = draw_boxes(frame, detections)
        out.write(annotated_frame)

    cap.release()
    out.release()
    os.remove(input_path)

    return FileResponse(output_path, media_type="video/mp4")