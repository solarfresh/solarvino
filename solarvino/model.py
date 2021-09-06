import cv2
import logging
import numpy as np
from typing import List
from . import output as output_parser
from .type import Detection


class BaseModel:
    def __init__(self, ie, model_xml, model_bin):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')
        self.net = ie.read_network(model=model_xml, weights=model_bin)
        self.image_blob_name, self.image_info_blob_name = self._get_inputs()
        self.n, self.c, self.h, self.w = self.net.inputs[self.image_blob_name].shape
        self.set_batch_size(1)

    def preprocess(self, inputs: np.array, meta=None):

        resized_image = cv2.resize(inputs, (self.w, self.h))

        meta = {'original_shape': inputs.shape,
                'resized_shape': resized_image.shape}

        h, w = resized_image.shape[:2]
        if h != self.h or w != self.w:
            resized_image = np.pad(resized_image, ((0, self.h - h), (0, self.w - w), (0, 0)),
                                   mode='constant', constant_values=0)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: resized_image}
        if self.image_info_blob_name:
            dict_inputs[self.image_info_blob_name] = [self.h, self.w, 1]
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in iter(self.net.inputs):
            new_shape = [batch] + self.net.inputs[input_layer].shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)

    def _get_inputs(self):
        image_blob_name = None
        image_info_blob_name = None
        for blob_name, blob in self.net.inputs.items():
            if len(blob.shape) == 4:
                image_blob_name = blob_name
            elif len(blob.shape) == 2:
                image_info_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name, image_info_blob_name


class KeypointModel(BaseModel):
    def __init__(self, ie, model_xml, model_bin):
        super().__init__(ie, model_xml, model_bin)
        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def postprocess(self, outputs, meta):
        original_shape = meta['original_shape']
        keypoints = self.output_parser(outputs)
        for kpt in keypoints.points:
            kpt.x = kpt.x * original_shape[1]
            kpt.y = kpt.y * original_shape[0]

        return keypoints

    def _get_output_parser(self, net, image_blob_name):
        try:
            parser = output_parser.FacialLandmark5Parser(net.outputs)
            self.logger.info('Use FacialLandmarkParser')
            return parser
        except ValueError as e:
            logging.debug(f'FacialLandmarkParser provides the message {e}')
            pass


class ObjectDetectionModel(BaseModel):
    def __init__(self,
                 ie,
                 model_xml,
                 model_bin,
                 labels: List[int],
                 confidence_score=.5,
                 nms_threshold=.45,):
        """
        :param labels: must be a list
        """
        super().__init__(ie, model_xml, model_bin)

        self.labels = labels
        self.confidence_score = confidence_score
        self.nms_threshold = nms_threshold

        self.image_blob_name, self.image_info_blob_name = self._get_inputs()
        self.n, self.c, self.h, self.w = self.net.inputs[self.image_blob_name].shape

        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def _get_output_parser(self, net, image_blob_name, bboxes='bboxes', labels='labels', scores='scores'):
        try:
            parser = output_parser.SingleOutputParser(net.outputs)
            self.logger.info('Use SingleOutputParser')
            return parser
        except ValueError as e:
            logging.debug(f'SingleOutputParser provides the message {e}')
            pass

        try:
            parser = output_parser.MultipleOutputParser(net.outputs, bboxes, scores, labels)
            self.logger.info('Use MultipleOutputParser')
            return parser
        except ValueError as e:
            logging.debug(f'MultipleOutputParser provides the message {e}')
            pass

        try:
            parser = output_parser.ScorePositionOutputParser(net.outputs, bboxes, scores)
            self.logger.info('Use ScorePositionOutputParser')
            return parser
        except ValueError as e:
            logging.debug(f'ScorePositionOutputParser provides the message {e}')
            pass

        try:
            parser = output_parser.BoxesLabelsParser(net.outputs, net.inputs[image_blob_name].shape[2:])
            self.logger.info('Use BoxesLabelsParser')
            return parser
        except ValueError as e:
            logging.debug(f'BoxesLabelsParser provides the message {e}')
            pass

        raise RuntimeError('Unsupported model outputs')

    def postprocess(self, outputs, meta):
        detections = self.output_parser(outputs)
        orginal_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']

        scale_x = self.w / resized_image_shape[1] * orginal_image_shape[1]
        scale_y = self.h / resized_image_shape[0] * orginal_image_shape[0]

        for detection in detections:
            detection.left *= scale_x
            detection.right *= scale_x
            detection.top *= scale_y
            detection.bottom *= scale_y

        detections = [detection for detection in detections if detection.score >= self.confidence_score]
        non_maximum_suppressed_detections = np.array([])
        for class_id in self.labels:
            class_non_maximum_suppressed_detections = self.nms(detections, class_id, threshold=self.nms_threshold)
            non_maximum_suppressed_detections = np.concatenate((non_maximum_suppressed_detections,
                                                               class_non_maximum_suppressed_detections))

        return non_maximum_suppressed_detections

    def nms(self, detections: List[Detection], class_id, threshold=.45):
        """
        todo: there are many copies, and it still can be reduced
        (top, left, bottom, right) => (y1, x1, y2, x2)
        """
        x1 = np.array([detection.left for detection in detections if detection.class_id == class_id])
        y1 = np.array([detection.top for detection in detections if detection.class_id == class_id])
        x2 = np.array([detection.right for detection in detections if detection.class_id == class_id])
        y2 = np.array([detection.bottom for detection in detections if detection.class_id == class_id])
        scores = np.array([detection.score for detection in detections if detection.class_id == class_id])
        class_detections = np.array([detection for detection in detections if detection.class_id == class_id])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep_indices = []
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return class_detections[keep_indices]
