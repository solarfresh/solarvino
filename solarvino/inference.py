import cv2
import os
import logging
import numpy as np
from typing import (Iterator, List)
from . import model
from .pipeline import (AsyncPipeline, OpenVINOConnector)
from .utils import dump_beautiful_json


class InferenceBase:
    # todo: probably we can extract common components with the class EvaluationBase
    """
    This function is designed following
    https://github.com/openvinotoolkit/open_model_zoo/blob/338630987b403a6981d03ab6d04c2d5ad367793a/
    demos/python_demos/object_detection_demo/object_detection_demo.py
    """
    model = None
    inference_pipeline = None
    meta_option = {}
    output_layer = None
    save_dir = ''

    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):
        raise NotImplementedError

    def inference(self, data_generator: Iterator, save_dir=None):
        logging.info('Inference is start...')
        self.save_dir = save_dir
        # prev_input_id = -1
        for input_id, image_path in data_generator:
            if self.inference_pipeline.callback_exceptions:
                raise self.inference_pipeline.callback_exceptions[0]

            if self.inference_pipeline.is_ready():
                inputs = cv2.imread(image_path)
                meta = {'image_path': image_path,
                        'image_array': inputs}
                meta.update(self.meta_option)
                self.inference_pipeline.submit_data(inputs, input_id, meta=meta)
            else:
                self.inference_pipeline.await_any()

            # self.get_infer_results(prev_input_id)
            # prev_input_id = input_id

            while self.inference_pipeline.has_completed_request():
                ids = self.inference_pipeline.completed_request_results.keys()
                for _id in list(ids):
                    self.get_infer_results(_id)

        return self

    def get_infer_results(self, input_id):
        results = self.inference_pipeline.get_result(input_id)
        if results:
            logging.debug(f'Inference results read {results}')
            self.infer_results_handler(results)
            return results

        return None

    def infer_results_handler(self, results):
        raise NotImplementedError


class HeadPoseInference(InferenceBase, OpenVINOConnector):
    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):
        self.model = model.HeadPoseModel(self.inference_engine,
                                         model_xml_path,
                                         model_bin_path)
        self.inference_pipeline = AsyncPipeline(self.inference_engine, self.model, self.plugin_config,
                                                device=self.device, num_infer_requests=self.num_infer_requests)
        return self

    def infer_results_handler(self, results):
        angle, meta = results
        if self.save_dir is not None:
            fname = os.path.basename(meta['image_path'])
            yaw = angle.yaw
            pitch = angle.pitch
            roll = angle.roll
            dump_beautiful_json({'fname': fname,
                                 'yaw': yaw,
                                 'pitch': pitch,
                                 'roll': roll}, os.path.join(self.save_dir, f'{os.path.splitext(fname)[0]}.json'))


class KeypointInference(InferenceBase, OpenVINOConnector):
    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):
        self.model = model.KeypointModel(self.inference_engine,
                                         model_xml_path,
                                         model_bin_path)
        self.inference_pipeline = AsyncPipeline(self.inference_engine, self.model, self.plugin_config,
                                                device=self.device, num_infer_requests=self.num_infer_requests)
        return self

    def infer_results_handler(self, results):
        keypoints, meta = results

        if self.save_dir is not None:
            image_array = meta['image_array']
            image_path = meta['image_path']

            annotation = {
                'annotations': {
                    "image_size": {
                        "depth": image_array.shape[2],
                        "height": image_array.shape[0],
                        "width": image_array.shape[1]
                    },
                    'keypoints': [],
                },
                "metadata": {
                    "class-map": {
                        "0": "left_eye",
                        "1": "right_eye",
                        "2": "nose_tip",
                        "3": "left_lip_corner",
                        '4': 'right_lip_corner'
                    }
                },
                "source-ref": os.path.basename(image_path)
            }

            for idx, point in enumerate(keypoints.points):
                annotation['annotations']['keypoints'].append({
                    "class_id": idx,
                    "x": int(point.x),
                    "y": int(point.y)
                })
                cv2.circle(image_array, (int(point.x), int(point.y)), radius=0, color=(0, 0, 255), thickness=3)
            image_save_path = os.path.join(self.save_dir, 'images', os.path.basename(meta['image_path']))
            cv2.imwrite(image_save_path, image_array)
            annotation_save_path = os.path.join(self.save_dir,
                                                'annotations',
                                                f'{os.path.splitext(os.path.basename(image_path))[0]}.json')
            dump_beautiful_json(annotation, annotation_save_path)


class ObjectDetectionInference(InferenceBase, OpenVINOConnector):
    default_attr = {}
    labels = []
    confidence_score = .5
    nms_threshold = .45

    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):

        self.default_attr.update(**kwargs)
        for key, value in self.default_attr.items():
            if key not in ['labels', 'confidence_score', 'nms_threshold']:
                continue

            self.__setattr__(key, value)

        self.model = model.ObjectDetectionModel(self.inference_engine,
                                                model_xml_path,
                                                model_bin_path,
                                                labels=self.labels,
                                                confidence_score=self.confidence_score,
                                                nms_threshold=self.nms_threshold,)
        self.inference_pipeline = AsyncPipeline(self.inference_engine, self.model, self.plugin_config,
                                                device=self.device, num_infer_requests=self.num_infer_requests)
        return self

    def infer_results_handler(self, results):
        detections, meta = results

        if self.save_dir is not None:
            image_array = meta['image_array']
            image_path = meta['image_path']
            fname = os.path.splitext(os.path.basename(image_path))[0]

            idx = -1
            for detection in detections:
                idx += 1
                sub_img = image_array[
                          int(detection.top):int(detection.bottom),
                          int(detection.left):int(detection.right)].copy()
                cv2.imwrite(os.path.join(self.save_dir, 'cropped', f'{fname}_{idx}.png'), sub_img.astype(np.uint8))

