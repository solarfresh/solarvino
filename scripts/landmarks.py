import os
import logging
from solarvino.iterator import DirectoryIterator
from solarvino.inference import KeypointInference


class Config:
    device = 'CPU'

    num_streams = ''
    num_threads = None
    num_infer_requests = 1

    image_directory = '/Users/huangshangyu/Downloads/experiment/maskimage/Crowd_Human/No_Mask'
    save_dir = '/Users/huangshangyu/Downloads/experiment/facekpt/tmp/inference'

    model_dir = '/Users/huangshangyu/Projects/solarfresh/solarvino/assets/intel/landmarks-regression-retail-0009'
    model_xml_path = os.path.join(model_dir, 'landmarks-regression-retail-0009.xml')
    model_bin_path = os.path.join(model_dir, 'landmarks-regression-retail-0009.bin')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    config = Config()

    data_generator = DirectoryIterator(directory=config.image_directory)
    inferencer = KeypointInference(device=config.device,
                                   num_streams=config.num_streams,
                                   num_threads=config.num_threads,
                                   num_infer_requests=config.num_infer_requests,)

    inferencer \
        .load_model(model_xml_path=config.model_xml_path,
                    model_bin_path=config.model_bin_path,) \
        .inference(data_generator=data_generator, save_dir=config.save_dir)
