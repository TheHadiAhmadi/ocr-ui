import paddleocr

class PaddleOCRDetector:
    def __init__(self):
        self.ocr = paddleocr.OCR()

    def predict(self, image_path):
        result = self.ocr.ocr(image_path)
        bboxes = [line[0] for line in result]
        return bboxes


# test the ocr and print the results. image is located at /dataset/images/1.jpeg
detector = PaddleOCRDetector()
bboxes = detector.predict('./dataset/images/1.jpeg')
print(bboxes)
