from numpy import *
from numpy import linalg as la
import numpy as np
import cv2
import time
def svd_denoise(img):
    u, sigma, vt = la.svd(img)
    h, w = img.shape[:2]
    h1 = int(h * 0.1)
    sigma1 = diag(sigma[:h1], 0)
    u1 = zeros((h, h1), float)
    u1[:, :] = u[:, :h1]
    vt1 = zeros((h1, w), float)
    vt1[:, :] = vt[:h1, :]
    return u1.dot(sigma1).dot(vt1)

img = cv2.imread("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/BM_DA/target/2018.07.04_22:05:22.257_7fc6e1023a200.png")
img = cv2.resize(img, (224, 224))
cv2.imshow("before", img)
cv2.waitKey(0)
st = time.time()
dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 10)
et = time.time()
print(et-st)
cv2.imshow("after", dst)
cv2.waitKey(0)
