import numpy as np
from .type import (Point, Detection, FacialLandmark5)


class OutputParserBase:
    @staticmethod
    def find_layer_by_name(name, layers):
        suitable_layers = [layer_name for layer_name in layers if name in layer_name]
        if not suitable_layers:
            raise ValueError('Suitable layer for "{}" output is not found'.format(name))

        if len(suitable_layers) > 1:
            raise ValueError('More than 1 layer matched to "{}" output'.format(name))

        return suitable_layers[0]


class FacialLandmark5Parser(OutputParserBase):
    def __init__(self, layers):
        if len(layers) != 1:
            raise ValueError('Network must have only one output.')
        self.output_name, output_data = next(iter(layers.items()))

    def __call__(self, outputs):
        kpts = outputs[self.output_name][0]
        kpts = kpts.reshape((-1, 2))
        points = [Point(kpt[0], kpt[1]) for kpt in kpts]
        return FacialLandmark5(points=points)


class SingleOutputParser(OutputParserBase):
    def __init__(self, all_outputs):
        if len(all_outputs) != 1:
            raise ValueError('Network must have only one output.')
        self.output_name, output_data = next(iter(all_outputs.items()))
        last_dim = np.shape(output_data)[-1]
        if last_dim != 7:
            raise ValueError('The last dimension of the output blob must be equal to 7, '
                             'got {} instead.'.format(last_dim))

    def __call__(self, outputs):
        return [Detection(left=xmin, top=ymin, right=xmax, bottom=ymax, score=score, class_id=label)
                for _, label, score, xmin, ymin, xmax, ymax in outputs[self.output_name][0][0]]


class ScorePositionOutputParser(OutputParserBase):
    def __init__(self, layers, bboxes_layer='bboxes', scores_layer='scores'):
        self.scores_layer = self.find_layer_by_name(scores_layer, layers)
        self.bboxes_layer = self.find_layer_by_name(bboxes_layer, layers)

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer][0]
        scores = outputs[self.scores_layer][0]
        return [Detection(*bbox, score=np.max(score), class_id=np.argmax(score)) for score, bbox in zip(scores, bboxes)]


class MultipleOutputParser(OutputParserBase):
    def __init__(self, layers, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.labels_layer = self.find_layer_by_name(labels_layer, layers)
        self.scores_layer = self.find_layer_by_name(scores_layer, layers)
        self.bboxes_layer = self.find_layer_by_name(bboxes_layer, layers)

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer][0]
        scores = outputs[self.scores_layer][0]
        labels = outputs[self.labels_layer][0]
        return [Detection(*bbox, score=score, class_id=label) for label, score, bbox in zip(labels, scores, bboxes)]


class BoxesLabelsParser(OutputParserBase):
    def __init__(self, layers, input_size, labels_layer='labels', default_label=0):
        try:
            self.labels_layer = self.find_layer_by_name(labels_layer, layers)
        except ValueError:
            self.labels_layer = None
            self.default_label = default_label

        self.bboxes_layer = self.find_layer_bboxes_output(layers)
        self.input_size = input_size

    @staticmethod
    def find_layer_bboxes_output(layers):
        filter_outputs = [name for name, data in layers.items() if len(np.shape(data)) == 2 and np.shape(data)[-1] == 5]
        if not filter_outputs:
            raise ValueError('Suitable output with bounding boxes is not found')
        if len(filter_outputs) > 1:
            raise ValueError('More than 1 candidate for output with bounding boxes.')
        return filter_outputs[0]

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer]
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        bboxes[:, 0::2] /= self.input_size[0]
        bboxes[:, 1::2] /= self.input_size[1]
        if self.labels_layer:
            labels = outputs[self.labels_layer]
        else:
            labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)

        detections = [Detection(*bbox, score=score, class_id=label) for label, score, bbox in zip(labels, scores, bboxes)]
        return detections
