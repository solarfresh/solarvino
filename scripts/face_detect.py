import os
import logging
from solarvino.iterator import DirectoryIterator
from solarvino.inference import ObjectDetectionInference


class Config:
    device = 'CPU'

    num_streams = ''
    num_threads = None
    num_infer_requests = 1

    image_directory = '/Users/huangshangyu/Downloads/experiment/sample_FLOW_AI_0604/images'
    save_dir = '/Users/huangshangyu/Downloads/experiment/facekpt/tmp/inference'

    model_dir = '/Users/huangshangyu/Projects/solarfresh/solarvino/assets/intel'
    model_xml_path = os.path.join(model_dir, 'face-detection-retail-0004', 'face-detection-retail-0004.xml')
    model_bin_path = os.path.join(model_dir, 'face-detection-retail-0004', 'face-detection-retail-0004.bin')

    class_map = {1: 1}
    nms_threshold = 0.45
    confidence_score = 0.5


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    config = Config()

    # data_generator = ObjectDetectionLabelGenerator()
    data_generator = DirectoryIterator(directory=config.image_directory)
    inferencer = ObjectDetectionInference(device=config.device,
                                          num_streams=config.num_streams,
                                          num_threads=config.num_threads,
                                          num_infer_requests=config.num_infer_requests,)

    inferencer \
        .load_model(model_xml_path=config.model_xml_path,
                    model_bin_path=config.model_bin_path,
                    labels=list(config.class_map.keys()),
                    confidence_score=config.confidence_score,
                    nms_threshold=config.nms_threshold,) \
        .inference(data_generator=data_generator,
                   save_dir=config.save_dir)
