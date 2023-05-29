from rknn.api import RKNN
import cv2
import numpy as np

# 图像后处理
import numpy as np
import cv2
import math
from nano import NanoDet


nanodet = NanoDet()

img = cv2.imread(r'/workspaces/rknn-toolkit/nanodet-main/111111111.jpg').astype(np.float32)
srcimg_ = img.copy()

img = nanodet.preprocess(img)

RKNN_PATH = r'/workspaces/rknn-toolkit/nanodet-service/nanodet-plus-m_416_torchscript.rknn'
# 创建 RKNN 对象
rknn = RKNN(True)
# 从当前目录加载 RKNN 模型 resnet_18
ret = rknn.load_rknn(path=RKNN_PATH)
if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
# 初始化运行时环境，设置目标开发板为 RK1808
# 如果只有一个设备，device_id 可以不填
ret = rknn.init_runtime()#target='rk1808', device_id='1808'
if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

# 调用 inference 接口进行推理
outputs = rknn.inference(inputs=[img],data_type='float32',data_format="nchw")[0]
print(outputs.shape)

results = nanodet.detect(outputs, 0.4, 0.4)

# center_priors = nanodet.generate_grid_center_priors(nanodet.input_size[0], nanodet.input_size[1], nanodet.strides)
# results = nanodet.decode_infer(outputs, center_priors, 0.4, 0.4)

for i in range(len(results)):
        bbox = results[i]
        cv2.rectangle(srcimg_, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=1)


for i in range(len(results)):
        r = results[i]
        print('label: ', nanodet.labels[r[5]])
        print('score: ', r[4])
        print('sigmoid_score: ', nanodet.sigmoid(r[4]))
        print('---------------')
cv2.imwrite(r'/workspaces/rknn-toolkit/nanodet-service/img2.png', srcimg_)
# cv2.imshow('result_img',src_img_)
# cv2.waitKey(0)
# cv2.destroyAllwindows()

# print(outputs[0].shape)
# print(outputs[0][0][0][:10])
# 所有待分类图片处理完后,释放 RKNN 对象
rknn.release()