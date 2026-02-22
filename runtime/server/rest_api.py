# depth_warp_vs/runtime/server/rest_api.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import torch
import yaml
from engine.inference import run_image_pair

app = FastAPI()

@app.post("/infer_concat")
async def infer_concat(cfg_path: str, file: UploadFile = File(...)):
    data = await file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    path = "upload.png"
    cv2.imwrite(path, img)
    out, aux = run_image_pair(cfg_path, concat=path, out="out.png")
    return {"output": "out.png"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
