# Clothes-Recognition-and-Retrieval
Clothes recognition and retrieval on clothing retails. Please see [report](.Final_report.pdf) for more details.

### Result on real data
![](./images/results.jpg)

## Usage
### Required package
1. tensorflow (2.0.0a0), GPU version can also be used
2. opencv (4.1.0.25)

### Installation
```bash
pip install -r requirements.txt
```

### Reproduce results
1. Run `main.py` to produce final classification result
2. Run `cloth_detection.py` to produce images with clothes detection bounding boxes.

### Model weights
We trained a Yolo-v3 object detection on [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, pre-trained model weights (tensorflow weights and darknet weights) can be download [here](https://drive.google.com/file/d/1DPydA0FpLYEHaFYDa8_oZAot_Ou5JefK/).

### Dataset
For the classifier, we use a relatively small dataset consists of only 46 clothes of 2 classes (clothes with stripes and clothes without stripes), the dataset can be download [here](https://drive.google.com/file/d/1oCMPB1MSsB3yJdOLm2iEZFGyYSKXQmIw/). 
![](./images/clothes_class.jpg)

## System pipeline
![](./images/system_pipeline.png)

## References