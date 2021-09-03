import cv2
import os
import logging
from typing import Iterator
from .model import KeypointModel
from .pipeline import (AsyncPipeline, OpenVINOConnector)


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

    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        raise NotImplementedError

    def get_infer_results(self, input_id):
        results = self.inference_pipeline.get_result(input_id)
        if results:
            logging.debug(f'Inference results read {results}')
            self.infer_results_handler(results)
            return results

        return None

    def infer_results_handler(self, results):
        raise NotImplementedError


class KeypointInference(InferenceBase, OpenVINOConnector):
    save_dir = None

    def load_model(self,
                   model_xml_path,
                   model_bin_path,
                   *args,
                   **kwargs):
        self.model = KeypointModel(self.inference_engine,
                                   model_xml_path,
                                   model_bin_path)
        self.inference_pipeline = AsyncPipeline(self.inference_engine, self.model, self.plugin_config,
                                                device=self.device, num_infer_requests=self.num_infer_requests)
        return self

    def inference(self, data_generator: Iterator, save_dir=None):
        logging.info('Inference is start...')
        self.save_dir = save_dir
        prev_input_id = -1
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

            self.get_infer_results(prev_input_id)
            prev_input_id = input_id

        return self

    def infer_results_handler(self, results):
        keypoints, meta = results

        if self.save_dir is not None:
            image_array = meta['image_array']
            for point in keypoints.points:
                cv2.circle(image_array, (int(point.x), int(point.y)), radius=0, color=(0, 0, 255), thickness=10)
            save_path = os.path.join(self.save_dir, os.path.basename(meta['image_path']))
            cv2.imwrite(save_path, image_array)
