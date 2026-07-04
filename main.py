from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=input_path,
        save=True,
        conf=0.4,
        project="/tmp/output",
        name="result"
    )

    output_dir = results[0].save_dir
    output_files = os.listdir(output_dir)
    output_path = os.path.join(output_dir, output_files[0])

    return FileResponse(output_path, media_type="video/mp4")