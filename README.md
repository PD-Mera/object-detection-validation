# Implement validation of object detection models

## Environments

Simply install requirements

``` bash
pip install -r requirements.txt
```

## Models

### YOLO Format

Model in onnx format, with input is an image with pixel values from [0, 1], outputs with shape `[batch_size, anchors, 4 + num_classes + 1]`

Example 1 line in outputs is

``` bash
# Format is [x_center, y_center, w, h, obj_conf, class_1_conf, class_2_conf]
[0.51 0.75 0.21 0.19 0.85 0.1 0.9]
```

### Retina Format

Model in onnx format, with input is an image read from `cv2.imread`, 3 outputs are `scores`, `bboxes` and `landmarks`

Example 1 line in outputs is

``` bash
# bboxes
[ 0.67413944  1.0143973  -1.1417826   0.16699778]
# scores
[9.9962950e-01 3.7046883e-04]
# landmarks
```

## Prepare labels

Ground truth labels in yolo format `[class_idx, x_center, y_center, w, h]`

``` txt
0 0.725000 0.287500 0.089394 0.070455
```

Predict labels also in yolo format, with confidence `[class_idx, x_center, y_center, w, h, conf]`

``` txt
0 0.8927168528238932 0.9185009002685547 0.17276369730631513 0.13056144714355478 0.8909046649932861
```

## Run

To get predict of model, modify path in `predict.py` and run

``` bash
python predict.py --onnx ./weights/yolov5m_v0_0_1.onnx \
                  --image_dir ../__testset__/TCB-VJA-testset_1/all/images \
                  --save_txt ./output_labels \
                  --backend yolo \
                  --gpu
```

To get evaluation value such as `mAP`, run

``` bash
python evaluate.py -gt ../__testset__/TCB-VJA-testset_1/all/labels \
                   -dr ./output_labels \
                   --image_dir ../__testset__/TCB-VJA-testset_1/all/images \
                   --num_classes 2
```

## References

- https://github.com/Cartucho/mAP
- https://github.com/ppogg/Retinaface_Ghost