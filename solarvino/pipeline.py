"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import threading
from collections import deque
from openvino.inference_engine import IECore


class AsyncPipeline:
    def __init__(self, ie, model, plugin_config, device='CPU', num_infer_requests=1):
        self.model = model
        self.logger = logging.getLogger()

        self.logger.info('Loading network to {} plugin...'.format(device))
        self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                        config=plugin_config, num_requests=num_infer_requests)

        self.empty_requests = deque(self.exec_net.requests)
        self.completed_request_results = {}
        self.callback_exceptions = []
        self.event = threading.Event()

    def inference_completion_callback(self, status, callback_args):
        request, id, meta, preprocessing_meta = callback_args
        try:
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: output for key, output in request.outputs.items()}
            self.completed_request_results[id] = (raw_outputs, meta, preprocessing_meta)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def submit_data(self, inputs, id, meta=None):
        """
        :param meta: (optional)
        """
        request = self.empty_requests.popleft()
        if len(self.empty_requests) == 0:
            self.event.clear()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta, preprocessing_meta))
        request.async_infer(inputs=inputs)

    def get_raw_result(self, id):
        if id in self.completed_request_results:
            return self.completed_request_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta = result
            return self.model.postprocess(raw_result, preprocess_meta), meta
        return None

    def is_ready(self):
        return len(self.empty_requests) != 0

    def has_completed_request(self):
        return len(self.completed_request_results) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()


class OpenVINOConnector:
    def __init__(self,
                 device='CPU',
                 num_streams='',
                 num_threads=None,
                 num_infer_requests=1):
        """
        :param device: Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
                       The sample will look for a suitable plugin for device specified. Default value is CPU.
        :param num_streams: Number of streams to use for inference on the CPU or/and GPU in throughput
                            mode (for HETERO and MULTI device cases use format
                            <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
        :param num_threads: Number of threads to use for inference on CPU (including HETERO cases).
        :param num_infer_requests: Number of infer requests
        """
        self.device = device
        self.num_streams = num_streams
        self.num_threads = num_threads
        self.num_infer_requests = num_infer_requests

        self.plugin_config = self._get_plugin_configs(self.device, self.num_streams, self.num_threads)
        # Load the Inference Engine API
        self.inference_engine = IECore()

    @staticmethod
    def _get_plugin_configs(device, num_streams, num_threads):
        config_user_specified = {}

        devices_nstreams = {}
        if num_streams:
            devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                if num_streams.isdigit() \
                else dict(device.split(':', 1) for device in num_streams.split(','))

        if 'MYRIAD' in device:
            config_user_specified['VPU_HW_STAGES_OPTIMIZATION'] = 'YES'

        if 'CPU' in device:
            if num_threads is not None:
                config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
            if 'CPU' in devices_nstreams:
                config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                    if int(devices_nstreams['CPU']) > 0 \
                    else 'CPU_THROUGHPUT_AUTO'

        if 'GPU' in device:
            if 'GPU' in devices_nstreams:
                config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                    if int(devices_nstreams['GPU']) > 0 \
                    else 'GPU_THROUGHPUT_AUTO'

        return config_user_specified
