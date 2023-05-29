import onnx
import yaml
import onnxruntime
import numpy as np
import cv2
import random
from nanodet.data.transform.warp import ShapeTransform
from nanodet.model.head import build_head
from nanodet.data.transform import Pipeline
from nanodet.util import Logger, cfg, load_config, load_model_weight
import torch

_COLORS = (
    np.array([0.000,0.447,0.741,0.850,0.325,0.098,0.929,0.694,0.125,0.494,0.184,0.556,0.466,0.674,0.188,0.301,0.745,0.933,0.635,0.078,0.184,0.300,0.300,0.300,0.600,0.600,0.600,
              1.000,0.000,0.000,1.000,0.500,0.000,0.749,0.749,0.000,0.000,1.000,0.000,0.000,0.000,1.000,0.667,0.000,1.000,0.333,0.333,0.000,0.333,0.667,0.000,0.333,1.000,0.000,
              0.667,0.333,0.000,0.667,0.667,0.000,0.667,1.000,0.000,1.000,0.333,0.000,1.000,0.667,0.000,1.000,1.000,0.000,0.000,0.333,0.500,0.000,0.667,0.500,0.000,1.000,0.500,
              0.333,0.000,0.500,0.333,0.333,0.500,0.333,0.667,0.500,0.333,1.000,0.500,0.667,0.000,0.500,0.667,0.333,0.500,0.667,0.667,0.500,0.667,1.000,0.500,1.000,0.000,0.500,
              1.000,0.333,0.500,1.000,0.667,0.500,1.000,1.000,0.500,0.000,0.333,1.000,0.000,0.667,1.000,0.000,1.000,1.000,0.333,0.000,1.000,0.333,0.333,1.000,0.333,0.667,1.000,
              0.333,1.000,1.000,0.667,0.000,1.000,0.667,0.333,1.000,0.667,0.667,1.000,0.667,1.000,1.000,1.000,0.000,1.000,1.000,0.333,1.000,1.000,0.667,1.000,0.333,0.000,0.000,
              0.500,0.000,0.000,0.667,0.000,0.000,0.833,0.000,0.000,1.000,0.000,0.000,0.000,0.167,0.000,0.000,0.333,0.000,0.000,0.500,0.000,0.000,0.667,0.000,0.000,0.833,0.000,
              0.000,1.000,0.000,0.000,0.000,0.167,0.000,0.000,0.333,0.000,0.000,0.500,0.000,0.000,0.667,0.000,0.000,0.833,0.000,0.000,1.000,0.000,0.000,0.000,0.143,0.143,0.143,
              0.286,0.286,0.286,0.429,0.429,0.429,0.571,0.571,0.571,0.714,0.714,0.714,0.857,0.857,0.857,0.000,0.447,0.741,0.314,0.717,0.741,0.50,0.5,0,]).astype(np.float32).reshape(-1, 3)
)

# 创建预测结果处理
load_config(cfg, r'./nanodet-plus-m_416.yml')
head_cfg = cfg.model.arch.head
head = build_head(head_cfg)

normalize = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]

def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img

# 画出结果
def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img

shape_transform = ShapeTransform(False)


raw_img = cv2.imread(r'./img1.png')
meta = {'img_info': {'id': [0], 'file_name': ['img.png'], 'height': [], 'width': []}, 'raw_img': [], 'img': [], 'warp_matrix': []}
meta['img'] = raw_img
img_meta = shape_transform(meta,[416,416])
img_img = img_meta["img"].astype(np.float32) / 255
img_img = _normalize(img_img, *normalize)
img = img_img.transpose(2, 0, 1)
img = img[np.newaxis,:]
img = np.round(img,4)

meta['img_info']['height'] = [raw_img.shape[0]]
meta['img_info']['width'] = [raw_img.shape[1]]
meta['raw_img'] = [raw_img]
meta['img'] = torch.tensor(img)
meta['warp_matrix'] = [meta['warp_matrix']]

print(img.shape)
sess = onnxruntime.InferenceSession(r'./nanodet-plus-m_416.onnx')
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run(None, {input_name:img.astype(np.float32)})[0]

pred_onx_tensor = torch.tensor(pred_onx)
# 处理结果
result = head.post_process(pred_onx_tensor,meta)[0]

result_img = overlay_bbox_cv(raw_img,result,cfg.class_names,0.5)

cv2.imshow('image',result_img)
cv2.waitKey(0)  #等待输入任何按键，
#当用户输入任何一个按键后即调用destroyAllWindows()关闭所有图像窗口
cv2.destroyAllWindows()

print(pred_onx)
print(np.argmax(pred_onx))
