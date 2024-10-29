from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import os
from dithering import *
import json
import threading
import asyncio
import uuid
import shutil
import svgwrite
from cdither import buffer_to_svg

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

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


# Global variables for progress tracking
progress = 0
processing_complete = False
dithered_images = {}
current_session_id = None

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "kernels": kernel_keys})

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None)
):
    global progress, processing_complete, dithered_images, current_session_id
    
    # Generate a new session ID
    current_session_id = str(uuid.uuid4())
    
    # Clear previous output
    shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    
    # Reset global variables
    progress = 0
    processing_complete = False
    dithered_images = {}

    filename = os.path.join(UPLOAD_FOLDER, f"{current_session_id}_{file.filename}")
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())
    
    # Start processing in a separate thread
    threading.Thread(target=process_image, args=(filename, width, height, current_session_id)).start()

    return StreamingResponse(progress_generator(), media_type="application/json")

@app.get("/output/{filename}")
async def display_image(filename: str):
    return FileResponse(os.path.join(OUTPUT_FOLDER, filename))

def process_image(image_path, width, height, session_id):
    global progress, processing_complete, dithered_images, current_session_id

    if session_id != current_session_id:
        return

    palette = [
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
        [255, 255, 0], # Yellow
        [255, 255, 255], # White
        [0, 0, 0]      # Black
    ]

    opts = {
        'palette': palette,
        'method': 2,
        'dithSerp': False
    }
    quant = RgbQuant(opts)

    input_image = Image.open(image_path)
    original_width, original_height = input_image.size
    
    # Adjust dimensions to maintain aspect ratio if only one is provided
    if width and not height:
        height = int((width / original_width) * original_height)
    elif height and not width:
        width = int((height / original_height) * original_width)
    elif not width and not height:
        width, height = original_width, original_height
    
    input_image = input_image.resize((width, height))
    
    total_kernels = len(kernel_keys)
    for index, dith_kern in enumerate(kernel_keys, 1):
        if session_id != current_session_id:
            return

        dithered_image_data = quant.reduce(input_image, dith_kern=dith_kern, dither_type="fast")
        
        # Convert dithered data to RGBA format
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                rgba_data[i, j, :3] = [
                    dithered_image_data[i * width + j] & 0xff,
                    (dithered_image_data[i * width + j] & 0xff00) >> 8,
                    (dithered_image_data[i * width + j] & 0xff0000) >> 16
                ]
                rgba_data[i, j, 3] = 255  # Set alpha channel to fully opaque
        
        output_filename = f'{session_id}_dithered_{dith_kern}.svg'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        buffer_to_svg(dithered_image_data, width, height, output_path)
        
        dithered_images[dith_kern] = output_filename

        # Update progress
        progress = int((index / total_kernels) * 100)

    processing_complete = True

async def progress_generator():
    global progress, processing_complete, dithered_images, current_session_id
    session_id = current_session_id
    while not processing_complete:
        if session_id != current_session_id:
            break
        yield json.dumps({'type': 'progress', 'progress': progress}) + '\n'
        await asyncio.sleep(0.1)  # Wait for 100ms before sending next update
    
    if session_id == current_session_id:
        yield json.dumps({'type': 'images', 'images': dithered_images}) + '\n'

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)