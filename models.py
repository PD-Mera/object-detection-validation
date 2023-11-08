import os
import numpy as np
import cv2
import onnxruntime
from tqdm import tqdm
from typing import List, Union, Tuple
from itertools import product as product
from math import ceil

class BasePredictor:
    def __init__(self):
        pass

    def image_reader(self, image_path: str) -> Tuple[np.ndarray, float, float]:
        pass

    def preprocess(self, image_cv: np.ndarray) -> np.ndarray:
        pass

    def postprocess(self, ort_outputs: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def __call__(self, 
                 image_path: str, 
                 visualize: Union[None, str] = None, 
                 save_txt: Union[None, str] = None):
        pass


class YOLOBasePredictor(BasePredictor):
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


class YOLOPredictor(YOLOBasePredictor):
    def __init__(self, 
                 onnx_path: str, 
                 onnx_providers: List[str] = ["CPUExecutionProvider"],
                 image_size: int = 960,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45) -> None:
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers = onnx_providers)
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def image_reader(self, image_path: str):
        inputs_cv = cv2.imread(image_path)
        inputs_cv = cv2.cvtColor(inputs_cv, cv2.COLOR_BGR2RGB)
        h, w, _ = inputs_cv.shape
        if h >= w:
            inputs_cv = self._image_resize(inputs_cv, height = self.image_size)
            inputs_cv = cv2.copyMakeBorder(inputs_cv, 0, 0, 0, self.image_size - inputs_cv.shape[1], cv2.BORDER_CONSTANT)
            h_scale = 1
            w_scale = h / w
        else:
            inputs_cv = self._image_resize(inputs_cv, width = self.image_size)
            inputs_cv = cv2.copyMakeBorder(inputs_cv, 0, self.image_size - inputs_cv.shape[0], 0, 0, cv2.BORDER_CONSTANT)
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
   
    def __call__(self, image_path: str, visualize: Union[None, str] = None, save_txt: Union[None, str] = None):
        inputs_cv, h_scale, w_scale = self.image_reader(image_path)
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


class RetinaBasePredictor():
    def __init__(self):
        pass

    def _decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def _decode_landm(self, pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms
    
    def _py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def _visualize(self, inputs_cv, outputs, h_scale, w_scale, save_path: str):
        h, w, _ = inputs_cv.shape
        for detected in outputs:
            x1 = int(detected[0] / w_scale)
            y1 = int(detected[1] / h_scale)
            x2 = int(detected[2] / w_scale)
            y2 = int(detected[3] / h_scale)
            inputs_cv = cv2.rectangle(inputs_cv, (x1, y1), (x2, y2), (0, 0, 255), 2) 

        # inputs_cv = cv2.cvtColor(inputs_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, inputs_cv)

    def _save_txt(self, inputs_cv, outputs, h_scale, w_scale, save_path):
        h, w, _ = inputs_cv.shape
        h = int(h / h_scale)
        w = int(w / w_scale)
        save_string = ""

        for detected in outputs:
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
            class_idx = 0
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w_ = x2 - x1
            h_ = y2 - y1
            save_string += f"{class_idx} {x_center} {y_center} {w_} {h_} {score}\n"

        with open(save_path, "w") as f:
            f.write(save_string)


class RetinaPredictor(RetinaBasePredictor):
    class PriorBox(object):
        def __init__(self, image_size=None):
            self.min_sizes = [[16, 32], [64, 128], [256, 512]]
            self.steps = [8, 16, 32]
            self.clip = True
            self.image_size = image_size
            self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
            self.name = "s"

        def forward(self):
            anchors = []
            for k, f in enumerate(self.feature_maps):
                min_sizes = self.min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]

            output = np.array(anchors).reshape(-1, 4)
            if self.clip:
                output.clip(max=1, min=0)
            return output
    
    def __init__(self, 
                 onnx_path: str, 
                 onnx_providers: List[str] = ["CPUExecutionProvider"],
                 image_size: int = 960,
                 conf_thres = 0.25,
                 iou_thres = 0.45) -> None:
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers = onnx_providers)
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def image_reader(self, image_path: str):
        inputs_cv = cv2.imread(image_path)
        return inputs_cv, 1.0, 1.0
    
    def preprocess(self, image_cv: np.ndarray) -> np.ndarray:
        img = np.float32(image_cv)

        # testing scale
        target_size = self.image_size
        max_size = self.image_size
        self.im_shape = img.shape
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])
        self.resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(self.resize * im_size_max) > max_size:
            self.resize = float(max_size) / float(im_size_max)
        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        self.im_height, self.im_width, _ = img.shape
        self.scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        self.scale_lm = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, ort_outputs: List[np.ndarray]):
        loc, conf, landms = ort_outputs
        priorbox = self.PriorBox(image_size=(self.im_height, self.im_width))
        priors = priorbox.forward()
        prior_data = priors
        boxes = self._decode(loc.squeeze(0), prior_data, [0.1, 0.2])
        boxes = boxes * self.scale / self.resize
        # boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0)[:, 1]
        landms = self._decode_landm(landms.squeeze(0), prior_data, [0.1, 0.2])
        
        landms = landms * self.scale_lm / self.resize
        # landms = landms.cpu().numpy()

        # ignore low scores
        
        inds = np.where(scores > self.conf_thres)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._py_cpu_nms(dets, self.iou_thres)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def __call__(self, image_path: str, visualize: Union[None, str] = None, save_txt: Union[None, str] = None):
        inputs_cv, h_scale, w_scale = self.image_reader(image_path)
        inputs = self.preprocess(inputs_cv)
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: np.dtype('float32').type(inputs),
        }
        ort_outputs = self.ort_session.run(None, ort_inputs)
        outputs = self.postprocess(ort_outputs)
        if visualize is not None:
            self._visualize(inputs_cv, outputs, h_scale, w_scale, save_path = visualize)
        if save_txt is not None:
            self._save_txt(inputs_cv, outputs, h_scale * self.resize, w_scale * self.resize, save_path = save_txt)
        return outputs