import math
import numpy as np
import cv2

class NanoDet():
        
        def __init__(self):
                self.input_size = [416, 416]
                self.has_gpu = True
                self.num_class = 80
                self.reg_max = 7
                self.strides = [8, 16, 32, 64]
                self.labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush"
                ]
                self.mean_vals = [103.53, 116.28, 123.675]
                self.norm_vals = [0.017429, 0.017507, 0.017125]
        
        def preprocess(self, mat):
                mat = cv2.resize(mat, (self.input_size[0], self.input_size[1]))
                mat = mat - self.mean_vals
                mat = mat * self.norm_vals
                mat = mat.transpose((2, 0, 1))
                mat = mat[np.newaxis,:].astype(np.float32)
                return mat
        
        def sigmoid(self, x):
                return (1 / (1 + math.exp(-x)))

        def generate_grid_center_priors(self, input_height, input_width, strides):                
                center_priors = []
                for i in range(len(strides)):
                        stride = strides[i]
                        feat_w = math.ceil(input_width / stride)
                        feat_h = math.ceil(input_height / stride)
                        for y in range(feat_h):
                                for x in range(feat_w):
                                        center_priors.append([x, y, stride])
                return center_priors

        def decode_infer(self,feats, center_priors, score_threshold, nms_threshold):
                results = []
                for i in range(80):
                        results.append([])
                num_points = len(center_priors)
                for idx in range(num_points):
                        ct_x, ct_y, stride = center_priors[idx]
                        scores = feats[0][idx]
                        score = 0
                        cur_label = 0
                        for label in range(self.num_class):
                                if scores[label] > score:
                                        score = scores[label]
                                        cur_label = label
                        if score > score_threshold:
                                bbox_pred = feats[0][idx][80:len(feats[0][idx])]
                                results[cur_label].append(self.disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride))
                        
                        dets = []
                        for i in range(len(results)):
                                self.nms(results[i], nms_threshold)
                                for j in range(len(results[i])):
                                        # results[i][j][4] = self.sigmoid(results[i][j][4])
                                        dets.append(results[i][j])


                return dets   

        def disPred2Bbox(self, bbox_pred, cur_label, cur_score, x, y, stride):
                ct_x = x * stride
                ct_y = y * stride
                dis_pred = [] 
                for i in range(4):
                        dis = 0
                        dis_after_sm = self.activation_function_softmax(bbox_pred[i * (self.reg_max + 1):i * (self.reg_max + 1) + 8], self.reg_max + 1)
                        for j in range(self.reg_max + 1):
                                dis += j * dis_after_sm[j]
                        dis *= stride
                        dis_pred.append(dis)
                xmin = max(ct_x - dis_pred[0], 0.0)
                ymin = max(ct_y - dis_pred[1], 0.0)
                xmax = min(ct_x + dis_pred[2], self.input_size[0])
                ymax = min(ct_y + dis_pred[3], self.input_size[1])
                return [xmin, ymin, xmax, ymax, cur_score, cur_label]

        def activation_function_softmax(self, src, length):
                alpha = np.max(src)
                denominator = 0
                dst = []
                for i in range(length):
                        dst.append(math.exp(src[i] - alpha))
                        denominator += dst[i]
                for i in range(length):
                        dst[i] /= denominator
                return dst

        def takeScore(self,element):
                return element.score
        
        def nms(self, input_boxes, nms_threshold):
                input_boxes.sort(key=lambda x:x[4], reverse=True)
                vArea = []
                for i in range(len(input_boxes)):
                        vArea.append((input_boxes[i][2] - input_boxes[i][0] + 1) * (input_boxes[i][3] - input_boxes[i][1] + 1))

                for j in range(len(input_boxes)):
                        for k in range(j + 1, len(input_boxes)):
                               xx1 = max(input_boxes[j][0], input_boxes[k][0]) 
                               yy1 = max(input_boxes[j][1], input_boxes[k][1]) 
                               xx2 = max(input_boxes[j][2], input_boxes[k][2]) 
                               yy2 = max(input_boxes[j][3], input_boxes[k][3])
                               w = max(0, xx2 - xx1 + 1)
                               h = max(0, yy2 - yy1 + 1)
                               inter = w * h
                               ovr = inter / (vArea[i] + vArea[j] - inter)
                               if ovr > nms_threshold:
                                        del input_boxes[k]
                                        del vArea[k]
                                        k -= 1

        def detect(self, outputs, score_threshold, nms_threshold):
                center_priors = self.generate_grid_center_priors(self.input_size[0], self.input_size[1], self.strides)
                results = self.decode_infer(outputs, center_priors, score_threshold, nms_threshold)
                return results
