from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import json
from pydantic import BaseModel


app = FastAPI()


class ProcessOptions(BaseModel):
    # add more image processing options here
    filename: str
    grayscale: bool = False
    contrast: int = 100
    resize_factor: float = 1
    binarize: bool = False
    denoise: bool = False

class SaveBoxesOptions(BaseModel):
    filename: str
    boxes: list

# read from json file DATA_PATH
DATA_PATH = "01-image-process/data.json"
SAMPLES_DIR = Path("static/samples")
PROCESSED_DIR = Path("static/processed")
IMAGES_DIR = Path("static/images")

try:
    with open(DATA_PATH, "r") as f:
        file = f.read()
        data = json.loads(file)
except FileNotFoundError:
    data = {}


# Ensure directories exist
SAMPLES_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Serve static files (CSS, JS, images)
app.mount("/app", StaticFiles(directory="01-image-process/front"), name="app")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def redirect_to_app():
    return RedirectResponse(url="/app/index.html")

@app.get("/list-images")
def list_sample_images():
    """List available images in the images folder"""
    images = [file.name for file in IMAGES_DIR.iterdir() if file.is_file()]
    print(images)
    return {"images": images}

@app.get("/list-processed")
def list_sample_processed():
    """List available processed in the processed folder"""
    processed = [file.name for file in PROCESSED_DIR.iterdir() if file.is_file()]
    print(processed)
    return {"images": processed}

@app.get("/process/{filename}")
def process_image_path(filename: str):
    return RedirectResponse(url="/app/process.html?filename=" + filename)
    

@app.get("/manual-ocr/{filename}")
def manual_ocr_image_path(filename: str):
    return RedirectResponse(url="/app/manual_ocr.html?filename=" + filename)
    

@app.get("/load-image/{filename}")
def load_image(filename: str):
    return JSONResponse(data[filename] if filename in data else {})


@app.post("/process")
def process_image(options: ProcessOptions):
    filename = options.filename
    # """Process a selected sample image (convert to grayscale and save to images/)"""
    input_path = IMAGES_DIR / filename
    output_path = PROCESSED_DIR / filename  # Save in images/ folder

    if not input_path.exists():
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    try:
        img = Image.open(input_path)

        if (options.grayscale or options.binarize):
            # Apply grayscale conversion
            img = img.convert("L")  # Convert to grayscale (L mode)
                # Convert image to grayscale before applying threshold
            image_cv = np.array(img)

            _, image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = Image.fromarray(image_cv)


        # Resize the img if resize factor is provided
        if options.resize_factor != 1.0:
            width, height = img.size
            new_width = int(width * options.resize_factor)
            new_height = int(height * options.resize_factor)
            img = img.resize((new_width, new_height))
        
        if (options.contrast):
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(img)
            contrast_factor = options.contrast / 100  # Convert contrast value to a factor
            img = enhancer.enhance(contrast_factor)


        # Apply denoising using OpenCV (Gaussian Blur or Median Filter)
        if options.denoise:
            # Convert to NumPy array for OpenCV processing
            image_cv = np.array(img)
            image_cv = cv2.GaussianBlur(image_cv, (5, 5), 0)
            # Convert back to PIL
            img = Image.fromarray(image_cv)
        
        if (options.binarize):
            image_cv = np.array(img)
            
            _, image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = Image.fromarray(image_cv)

                
        # Save the updated image back to the static folder
        img.save(output_path)
        
        data[filename] = options.model_dump()
        # save options to file
        print(data)
        with open(DATA_PATH, "w") as f:
            f.write(json.dumps(data))

        return {"message": "Image successfully converted."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    

@app.post("/save-boxes")
def save_boxes(options: SaveBoxesOptions):
    filename = options.filename
    # """Process a selected sample image (convert to grayscale and save to images/)"""
    boxes = options.boxes
 
    # fix this:

    if filename not in data:
        data[filename] = {
            "filename": filename,
        }
    
    data[filename].update({
        "boxes": boxes
    })
        
    with open(DATA_PATH, "w") as f:
        f.write(json.dumps(data))

    return {"message": "Boxes saved successfully.", "success": True}


@app.get("/get-image/{filename}")
def get_processed_image(filename: str):
    """Retrieve a processed image from the images folder"""
    file_path = IMAGES_DIR / filename
    if not file_path.exists():
        return JSONResponse(content={"error": "Processed file not found"}, status_code=404)
    return FileResponse(file_path)
