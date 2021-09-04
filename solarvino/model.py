import cv2
import logging
import numpy as np
from solarvino.output import FacialLandmark5Parser


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
            parser = FacialLandmark5Parser(net.outputs)
            self.logger.info('Use FacialLandmarkParser')
            return parser
        except ValueError as e:
            logging.debug(f'FacialLandmarkParser provides the message {e}')
            pass
