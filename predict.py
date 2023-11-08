import argparse
import os
import numpy as np
import cv2
import onnxruntime
from tqdm import tqdm
from typing import List, Union
from models import YOLOPredictor, RetinaPredictor

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help="onnx model path", type=str)
    parser.add_argument('--backend', help="onnx model path", type=str, default="yolo", choices=["yolo", "retina"])
    parser.add_argument('--image_dir', help="path to validation image folder", type=str)
    parser.add_argument('--save_txt', help="path to save predicted txt", type=str, default="./output_labels")
    parser.add_argument('--gpu', help="use onnxruntime-gpu", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    if args.gpu:
        onnx_providers = ["CUDAExecutionProvider"]
    else:
        onnx_providers = ["CPUExecutionProvider"]
    
    if args.backend == "yolo":
        predictor = YOLOPredictor(f'{args.onnx}', onnx_providers = onnx_providers, image_size = 960)
    if args.backend == "retina":
        predictor = RetinaPredictor(f'{args.onnx}', onnx_providers = onnx_providers, image_size = 640)

    os.makedirs(args.save_txt, exist_ok=True)
    SAVE_LABEL_PATH = args.save_txt

    for image_name in tqdm(os.listdir(args.image_dir)):
        img_path = os.path.join(args.image_dir, image_name)

        predictor(image_path = img_path,
                  visualize = os.path.join(SAVE_LABEL_PATH, image_name),
                  save_txt = os.path.join(SAVE_LABEL_PATH, image_name.replace(".jpg", ".txt"))
                  )
        