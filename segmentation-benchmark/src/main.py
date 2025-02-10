import json
# Main file
# import tools.paddle_detector
from tools.tesseract_detector import TesseractDetector
from tools.utilities import load_dataset, run_benchmarks

# load coco file

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")

def main():
    metrics = {}
    detectors = [TesseractDetector("tesseract"), TesseractDetector("another_tesseract")]
    for detector in detectors:
        metrics[detector.name] = run_benchmarks(detector, dataset)

    with(open('./segmentation-benchmark/output/metrics.json', "w") as f):
        f.write(json.dumps(metrics, indent=4))

    print("Benchmarks saved to segmentation-benchmark/output folder")


main()