import cv2
import numpy as np

img = cv2.imread(r'/workspaces/rknn-toolkit/nanodet-main/111111111.jpg')
img = cv2.resize(img,(416,416))

print(img[0][0],img[1][1])
mean = [103.53, 116.28, 123.675]
img = img - mean
print(img[0][0],img[1][1])
norm_vals = [0.017429, 0.017507, 0.017125]
img = img * norm_vals
print(img[0][0],img[1][1])
print(img.shape)
img = img.transpose((2,0,1))
print(img.shape)
img = img[ np.newaxis,:]
print(img)
print(img.shape)
# cv2.imwrite(r'/workspaces/rknn-toolkit/nanodet-main/111111111.jpg',img)