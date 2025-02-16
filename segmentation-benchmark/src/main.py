import json
# Main file
# import tools.paddle_detector
from tools.tesseract_detector import TesseractDetector
from tools.easyocr_detector import EasyocrDetector
from tools.utilities import load_dataset, run_benchmarks

# load coco file

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")

tesseract1 = TesseractDetector('tesseract1')
tesseract2 = TesseractDetector('tesseract2')
tesseract3 = TesseractDetector('tesseract3')

easyocr1 = EasyocrDetector("easyocr 0.2")
easyocr2 = EasyocrDetector("easyocr 0.5")
easyocr3 = EasyocrDetector("easyocr 0.8")

tesseract1.threshold = 0.2
tesseract2.threshold = 0.5
tesseract3.threshold = 0.8

easyocr1.threshold = 0.2
easyocr2.threshold = 0.5
easyocr3.threshold = 0.8

def main():
    metrics = {}
    detectors = [tesseract1, easyocr1, easyocr2, easyocr3]

    for detector in detectors:
        metrics[detector.name] = run_benchmarks(detector, dataset)

    with(open('./segmentation-benchmark/output/metrics.json', "w") as f):
        f.write(json.dumps(metrics, indent=4))

    print("Benchmarks saved to segmentation-benchmark/output folder")


main()
