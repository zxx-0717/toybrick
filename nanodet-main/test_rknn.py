from rknn.api import RKNN
import torch
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


# logger = Logger(-1, r'./', False)
# model = build_model(r'./nanodet-plus-m_416.yml')
# checkpoint = torch.load(r'./nanodet-plus-m_416_checkpoint.ckpt', map_location=lambda storage, loc: storage)
# load_model_weight(model, checkpoint, logger)

rk = RKNN(True)

rk.config()

# rk.load_pytorch(r'./nanodet-plus-m_416.onnx')

rk.load_onnx(r'./nanodet-plus-m_416.onnx')

rk.build(False)

rk.export_rknn(r'./nanodet-plus-m_416.rknn')
