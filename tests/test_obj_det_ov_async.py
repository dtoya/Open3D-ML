import os
#import open3d.ml as _ml3d
import ml3d as _ml3d
#import open3d.ml.torch as ml3d
import ml3d.torch as ml3d
from tqdm import tqdm

import time

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model, device='cpu')
model = ml3d.models.OpenVINOModel(model)
model.to('cpu')
model.set_async_mode()

cfg.dataset['dataset_path'] = "../dataset/KITTI"
dataset = _ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.

num_images = 100
start = time.perf_counter()

for idx in tqdm(range(num_images)):
    data = test_split.get_data(idx)
    pipeline.submit_inference(data)
    while True:
        result = pipeline.get_result()
        if result == None:
            break

pipeline.wait_all()

end = time.perf_counter()
time_ir = end - start
print(
    f"IR model in Inference Engine/CPU: {time_ir/num_images:.4f} \n"
    f"seconds per image, FPS (includes pre-process): {num_images/time_ir:.2f}"
)


