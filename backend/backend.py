# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import os
import json
import threading
import asyncio
import uuid
import shutil
from typing import Optional
from dithering import *
from cdither import buffer_to_svg

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

# Create necessary folders
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Define the dithering kernels
kernel_keys = [
    'FloydSteinberg',
    'Atkinson',
    'Sierra24A',
    'Fan',
    'ShiauFan',
    'ShiauFan2',
    'JarvisJudiceNinke',
    'Stucki',
    'Burkes',
    'Sierra3',
    'Sierra2'
]

# Global state management
class ProcessingState:
    def __init__(self):
        self.progress = 0
        self.processing_complete = False
        self.dithered_images = {}
        self.current_session_id = None

state = ProcessingState()

@app.get("/api/kernels")
async def get_kernels():
    return {"kernels": kernel_keys}

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None)
):
    # Generate new session ID and reset state
    state.current_session_id = str(uuid.uuid4())
    state.progress = 0
    state.processing_complete = False
    state.dithered_images = {}
    
    # Clear and recreate output directory
    shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    
    # Save uploaded file
    filename = os.path.join(UPLOAD_FOLDER, f"{state.current_session_id}_{file.filename}")
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())
    
    # Start processing in background
    threading.Thread(
        target=process_image,
        args=(filename, width, height, state.current_session_id)
    ).start()
    
    return {"session_id": state.current_session_id}

@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    if session_id != state.current_session_id:
        return {"error": "Invalid session"}
    
    return {
        "progress": state.progress,
        "complete": state.processing_complete,
        "images": state.dithered_images if state.processing_complete else None
    }

@app.get("/api/output/{filename}")
async def get_output(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

def process_image(image_path: str, width: Optional[int], height: Optional[int], session_id: str):
    if session_id != state.current_session_id:
        return

    palette = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 255, 255],# White
        [0, 0, 0]       # Black
    ]

    opts = {
        'palette': palette,
        'method': 2,
        'dithSerp': False
    }
    quant = RgbQuant(opts)

    # Process image
    input_image = Image.open(image_path)
    original_width, original_height = input_image.size
    
    # Calculate dimensions
    if width and not height:
        height = int((width / original_width) * original_height)
    elif height and not width:
        width = int((height / original_height) * original_width)
    elif not width and not height:
        width, height = original_width, original_height
    
    input_image = input_image.resize((width, height))
    
    # Process each kernel
    total_kernels = len(kernel_keys)
    for index, dith_kern in enumerate(kernel_keys, 1):
        if session_id != state.current_session_id:
            return

        dithered_image_data = quant.reduce(input_image, dith_kern=dith_kern, dither_type="fast")
        
        # Save output
        output_filename = f'{session_id}_dithered_{dith_kern}.svg'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        buffer_to_svg(dithered_image_data, width, height, output_path)
        
        state.dithered_images[dith_kern] = output_filename
        state.progress = int((index / total_kernels) * 100)

    state.processing_complete = True

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)