import cv2
import pytesseract

class TesseractOCR:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def run(self, image_path):
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")
        
        # Convert the image to RGB (Tesseract expects RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use pytesseract to do OCR on the image
        data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)

        # Prepare an array of text and confidence
        result = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])
            if text:  # Only include text that is not empty
                result.append({'text': text, 'confidence': confidence})

        return result
