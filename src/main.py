from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Detect objects in images using YOLOv8 and return annotated results",
    version="1.0.0"
)

# Mount static directory for image results
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  

# Create results directory
Path("static/results").mkdir(parents=True, exist_ok=True)

@app.post("/detect/", tags=["Object Detection"])
async def detect_objects(
    image: UploadFile = File(..., description="Image file to process")
):
    """Detect objects in image using YOLOv8 and return annotated image"""
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLOv8 detection
        results = model.predict(img, conf=0.5)
        
        # Annotate image with results
        annotated_img = results[0].plot()
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"static/results/{timestamp}_{image.filename}"
        cv2.imwrite(output_path, annotated_img)
        return FileResponse(
              f"static/results/{timestamp}_{image.filename}",
              media_type="image/png")  # or image/jpeg based on your file

        # return {
        #     "filename": image.filename,
        #     "detected_objects": len(results[0].boxes),
        #     "result_url": f"/static/results/{timestamp}_{image.filename}"
        # }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Status"])
def health_check():
    return {
        "status": "active", 
        "model": "yolov8n.pt",
        "endpoints": {
            "detect": "POST /detect/",
            "result": "GET /static/results/{filename}"
        }
    }
