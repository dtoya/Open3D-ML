import io
import copy

import torch

import openvino.runtime as ov 
import queue

from .. import dataloaders


def pointpillars_extract_feats(self, x):
    x = self.backbone(x)
    x = self.neck(x)
    return x


class OpenVINOModel:
    """Class providing OpenVINO backend for PyTorch models.

    OpenVINO model is initialized from ONNX representation of PyTorch graph.
    """

    def __init__(self, base_model):
        self.core = ov.Core()
        self.compiled_model = None
        self.base_model = base_model
        self.device = "CPU"
        self.ireqs = None
        self.async_mode = False 
        self.results = None 

        # A workaround for unsupported torch.square by ONNX
        torch.square = lambda x: torch.pow(x, 2)

    def _get_input_names(self, inputs):
        names = []
        for name, tensor in inputs.items():
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    names.append(name + str(i))
            else:
                names.append(name)
        return names

    def _get_inputs(self, inputs, export=False):
        if isinstance(inputs, dataloaders.concat_batcher.KPConvBatch):
            inputs = {
                'features': inputs.features,
                'points': inputs.points,
                'neighbors': inputs.neighbors,
                'pools': inputs.pools,
                'upsamples': inputs.upsamples,
            }
        elif isinstance(inputs, dataloaders.concat_batcher.ObjectDetectBatch):
            voxels, num_points, coors = self.base_model.voxelize(inputs.point)
            voxel_features = self.base_model.voxel_encoder(
                voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            x = self.base_model.middle_encoder(voxel_features, coors,
                                               batch_size)

            inputs = {
                'x': x,
            }
        elif not isinstance(inputs, dict):
            raise TypeError(f"Unknown inputs type: {inputs.__class__}")
        return inputs

    def set_async_mode(self, value=True):
        self.async_mode = value 

    def _callback_function(self, request, userdata):
        output = request.results
        if len(output) == 1:
            output = next(iter(output.values()))
            output = torch.tensor(output)
        else:
            output = tuple([torch.tensor(out) for out in output.values()])
        result = {'input': userdata, 'output': output}
        self.results.put(result)
    
    def get_results(self):
        return self.results 
    
    def wait_all(self):
        self.ireqs.wait_all()

    def _read_torch_model(self, inputs):
        inputs = copy.deepcopy(inputs)
        tensors = self._get_inputs(inputs)
        input_names = self._get_input_names(tensors)

        # Forward origin inputs instead of export <tensors>
        origin_forward = self.base_model.forward
        self.base_model.forward = lambda x: origin_forward(inputs)
        self.base_model.extract_feats = lambda *args: pointpillars_extract_feats(
            self.base_model, tensors[input_names[0]])

        buf = io.BytesIO()
        self.base_model.device = torch.device('cpu')
        self.base_model.eval()
        torch.onnx.export(self.base_model,
                          tensors,
                          buf,
                          input_names=input_names)

        self.base_model.forward = origin_forward

        net = self.core.read_model(buf.getvalue())
        config = {
            'PERFORMANCE_HINT': 'THROUGHPUT',
            'NUM_STREAMS': 'AUTO'
            }
        self.compiled_model = self.core.compile_model(net, str(self.device).upper(), config)
        if self.async_mode:
            nireq = self.compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            num_streams = self.compiled_model.get_property("NUM_STREAMS")
            print("nireq = {}".format(nireq))
            print("num_streams = {}".format(num_streams))
            self.ireqs = ov.AsyncInferQueue(self.compiled_model)
            self.ireqs.set_callback(self._callback_function)
            self.results = queue.Queue()

    def forward(self, inputs):
        if self.compiled_model is None:
            self._read_torch_model(inputs)

        inputs_orig = inputs
        inputs = self._get_inputs(inputs)

        tensors = {}
        for name, tensor in inputs.items():
            if name == 'labels':
                continue
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    if tensor[i].nelement() > 0:
                        tensors[name + str(i)] = tensor[i].detach().numpy()
            else:
                if tensor.nelement() > 0:
                    tensors[name] = tensor.detach().numpy()

        output = None
        if self.ireqs:
            idle_id = self.ireqs.get_idle_request_id()
            self.ireqs.start_async(tensors, inputs_orig)
        else:
            output = self.compiled_model(tensors)
            if len(output) == 1:
                output = next(iter(output.values()))
                output = torch.tensor(output)
            else:
                output = tuple([torch.tensor(out) for out in output.values()])

        return output

    def __call__(self, inputs):
        return self.forward(inputs)

    def load_state_dict(self, *args):
        self.base_model.load_state_dict(*args)

    def eval(self):
        pass

    @property
    def cfg(self):
        return self.base_model.cfg

    @property
    def classes(self):
        return self.base_model.classes

    def inference_end(self, *args):
        return self.base_model.inference_end(*args)

    def preprocess(self, *args):
        return self.base_model.preprocess(*args)

    def transform(self, *args):
        return self.base_model.transform(*args)

    def to(self, device):
        self.device = device
