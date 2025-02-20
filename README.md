# Ocr benchmarks

1. to change dataset you need to add images to segmentation-benchmark/dataset/images folder.
2. to annotate images you can use roboflow.com site and copy annotated data(json) to segmentation-benchmark/dataset/annotations/{file_name}.json
3. to generate coco based dataset, run 
```
python segmentation-benchmark/scripts/generate_coco.py
```
it will create segmentation-benchmark/dataset/coco.json file

4. to run benchmarks, you can run
```
python segmentation-benchmark/src/main.py
```
output will be generated in segmentation-benchmark/output folder.


to generate html of outputs, run 
```
python segmentation-benchmark/scripts/generate_output.py
```
segmentation-benchmark/output/metrics_comparison.html file will be created which you can open in browser.

