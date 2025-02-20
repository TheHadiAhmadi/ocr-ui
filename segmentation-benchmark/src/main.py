import json
# Main file
# import tools.paddle_detector
from tools.tesseract_detector import TesseractDetector
from tools.easyocr_detector import EasyocrDetector
from tools.paddle_detector import PaddleDetector
from tools.utilities import load_dataset, run_benchmarks

# load coco file

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")

tesseract1 = TesseractDetector('tesseract1')
paddle1 = PaddleDetector('paddle1')
easyocr1 = EasyocrDetector("easyocr 0.2")
# easyocr2 = EasyocrDetector("easyocr 0.5")
# easyocr3 = EasyocrDetector("easyocr 0.8")

tesseract1.threshold = 0.2

easyocr1.threshold = 0.2
# easyocr2.threshold = 0.5
# easyocr3.threshold = 0.8

def main():
    metrics = {}
    # detectors = [tesseract1, easyocr1, paddle1]
    detectors = [tesseract1]

    for detector in detectors:
        metrics[detector.name] = run_benchmarks(detector, dataset)

    with(open('./segmentation-benchmark/output/metrics.json', "w") as f):
        f.write(json.dumps(metrics, indent=4))

    print("Benchmarks saved to segmentation-benchmark/output folder")


main()
