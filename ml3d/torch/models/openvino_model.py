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
        self.ie = ov.Core()
        self.exec_net = None
        self.base_model = base_model
        self.device = "CPU"
        self.ireqs = None
        self.async_mode = False 
        self.results = queue.Queue()

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

    def set_async_mode(self):
        self.async_mode = True

    def _callback_function(self, request, userdata):
        #output = request.get_output_tensor()
        output = request.get_output_tensor().data
        if len(output) == 1:
            output = next(iter(output.values()))
            output = torch.tensor(output)
        else:
            output = tuple([torch.tensor(out) for out in output.values()])
        result = {'input': userdata, 'output': output}
        self.results.put(result)
    
    def get_result(self):
        return self.results 

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

        #net = self.ie.read_model(buf.getvalue(), b'', init_from_buffer=True)
        net = self.ie.read_model(buf.getvalue(), b'')
        tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}
        self.exec_net = self.ie.compile_model(net, str(self.device).upper(), tput)
        if self.async_mode:
            self.ireqs = ov.AsyncInferQueue(self.exec_net)
            self.ireqs.set_callback(self._callback_function)

    def forward(self, inputs):
        if self.exec_net is None:
            self._read_torch_model(inputs)

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
        if self.async_mode:
            idle_id = self.ireqs.get_idle_request_id()
            self.ireqs.start_async(tensors, tensors)
        else:
            output = self.exec_net(tensors)

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
