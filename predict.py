import os
import numpy as np
import cv2
import onnxruntime
from tqdm import tqdm
from typing import List, Union

class BasePredictor:
    def __init__(self):
        pass

    def __box_iou(self, box1, box2, eps=1e-7):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (np.ndarray[N, 4])
            box2 (np.ndarray[M, 4])
        Returns:
            iou (np.ndarray[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        a1, a2 = np.split(box1, 2, axis=1)
        b1, b2 = np.split(box2, 2, axis=1)
        inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(2)

        eps_arr = np.full_like(inter, eps)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps_arr)

    def __xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def __numpy_nms(self, boxes, scores, threshold):
        # Ensure there are boxes to suppress
        if len(boxes) == 0:
            return []

        # Sort boxes by their scores in descending order
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            # Select the box with the highest score
            i = order[0]
            keep.append(i)

            # Calculate the IoU (Intersection over Union) with other boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            intersection = w * h
            union = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - intersection
            iou = intersection / union

            # Find the boxes to suppress (IoU > threshold)
            suppressed = np.where(iou <= threshold)[0]
            order = order[suppressed + 1]

        return keep

    def _image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def _non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            multi_label=False,
            labels=(),
            max_det=300,
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        mi = 5 + nc  # mask start index
        output = [np.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.__xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                # x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
                x = np.concatenate((box[i], x[i, 5 + j][:, None], j[:, None].astype(float), mask[i]), axis=1)
            else:  # best class only
                # conf, j = x[:, 5:mi].max(1, keepdim=True)
                conf = np.max(x[:, 5:mi], axis=1, keepdims=True)
                j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
                # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
                x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.flatten() > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[np.any(x[:, 5:6] == classes, axis=1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
            sorted_indices = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[sorted_indices]

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            # i = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_thres)
            i = self.__numpy_nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.__box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                x[i, :4] = np.dot(x[:, :4], weights.T).astype(float) / np.sum(weights, axis=1, keepdims=True)
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output
    
    def _visualize(self, inputs_cv, outputs, h_scale, w_scale, save_path: str):
        size = inputs_cv.shape[0]
        for detected in outputs[0]:
            x1 = int(detected[0])
            y1 = int(detected[1])
            x2 = int(detected[2])
            y2 = int(detected[3])
            inputs_cv = cv2.rectangle(inputs_cv, (x1, y1), (x2, y2), (0, 0, 255), 2) 

        inputs_cv = cv2.cvtColor(inputs_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, inputs_cv[0:int(size / h_scale), 0:int(size / w_scale)])

    def _save_txt(self, inputs_cv, outputs, h_scale, w_scale, save_path):
        size = inputs_cv.shape[0]
        h = int(size / h_scale)
        w = int(size / w_scale)
        save_string = ""

        for detected in outputs[0]:
            x1 = float(detected[0]) / w
            y1 = float(detected[1]) / h
            x2 = float(detected[2]) / w
            y2 = float(detected[3]) / h
            if x1 < 0.0:
                x1 = 0.0
            if y1 < 0.0:
                y1 = 0.0
            if x2 > 1.0:
                x2 = 1.0
            if y2 > 1.0:
                y2 = 1.0
            score = float(detected[4])
            class_idx = int(detected[5])
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w_ = x2 - x1
            h_ = y2 - y1
            save_string += f"{class_idx} {x_center} {y_center} {w_} {h_} {score}\n"

        with open(save_path, "w") as f:
            f.write(save_string)


class Predictor(BasePredictor):
    def __init__(self, 
                 onnx_path: str, 
                 onnx_providers: List[str] = ["CPUExecutionProvider"],
                 conf_thres = 0.25,
                 iou_thres = 0.45) -> None:
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers = onnx_providers)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def image_reader(self, image_path: str, image_size: int = 960):
        inputs_cv = cv2.imread(image_path)
        inputs_cv = cv2.cvtColor(inputs_cv, cv2.COLOR_BGR2RGB)
        h, w, _ = inputs_cv.shape
        if h >= w:
            inputs_cv = self._image_resize(inputs_cv, height = image_size)
            inputs_cv = cv2.copyMakeBorder(inputs_cv, 0, 0, 0, image_size - inputs_cv.shape[1], cv2.BORDER_CONSTANT)
            h_scale = 1
            w_scale = h / w
        else:
            inputs_cv = self._image_resize(inputs_cv, width = image_size)
            inputs_cv = cv2.copyMakeBorder(inputs_cv, 0, image_size - inputs_cv.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            h_scale = w / h
            w_scale = 1
        return inputs_cv, h_scale, w_scale

    def preprocess(self, image_cv: np.ndarray):
        inputs = image_cv / 255.0
        # inputs -= [0.485, 0.456, 0.406]
        # inputs /= [0.229, 0.224, 0.225]   
        inputs = inputs.transpose(2, 0, 1)
        inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def postprocess(self, ort_outputs: List[np.ndarray]):
        outputs = self._non_max_suppression(ort_outputs[0], conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        return outputs
   
    def __call__(self, image_path: str, image_size: int = 960, visualize: Union[None, str] = None, save_txt: Union[None, str] = None):
        inputs_cv, h_scale, w_scale = self.image_reader(image_path, image_size)
        inputs = self.preprocess(inputs_cv)
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: np.dtype('float32').type(inputs),
        }
        ort_outputs = self.ort_session.run(None, ort_inputs)
        outputs = self.postprocess(ort_outputs)
        if visualize is not None:
            self._visualize(inputs_cv, outputs, h_scale, w_scale, save_path = visualize)
        if save_txt is not None:
            self._save_txt(inputs_cv, outputs, h_scale, w_scale, save_path = save_txt)
        return outputs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help="onnx model path", type=str)
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
    predictor = Predictor(f'{args.onnx}', onnx_providers = onnx_providers)

    os.makedirs(args.save_txt, exist_ok=True)
    SAVE_LABEL_PATH = args.save_txt

    for image_name in tqdm(os.listdir(args.image_dir)):
        img_path = os.path.join(args.image_dir, image_name)

        predictor(image_path = img_path,
                #   visualize = os.path.join(SAVE_LABEL_PATH, image_name),
                  save_txt = os.path.join(SAVE_LABEL_PATH, image_name.replace(".jpg", ".txt")))
        