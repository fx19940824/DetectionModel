import cv2
import numpy as np
import glob
path1 = "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/demo/2019.02.27_17:27:02.844_7f57c40009c0_21.png"
# path3 = "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/demo/2019.03.01_22:42:40.250_7f9d500009c0_20.png"
# path2 = "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/demo/2019.02.27_10:08:54.426_7f731c0009c0_39.png"


# imgs = glob.glob("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_res_seg/27_处理/多球少球/*.png")
#
# kernel = np.ones((3, 3), np.uint8)
# for path in imgs:
#     img = cv2.imread(path)
#     img = cv2.GaussianBlur(img, (5,5), 0)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, justthresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
#     cv2.imshow("justthresh", justthresh)
#     _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
#     gray_laplacian = cv2.Laplacian(gray, -1, ksize=5)
#     cv2.imshow("gray_laplacian_", gray_laplacian)
#     # cv2.waitKey(0)
#     _, gray_laplacian = cv2.threshold(gray_laplacian, 120, 255, cv2.THRESH_BINARY)
#     gray_laplacian = cv2.morphologyEx(gray_laplacian, cv2.MORPH_OPEN, kernel, 3)
#     # cv2.imshow("gray_laplacian", gray_laplacian)
#     # cv2.waitKey(0)
#     mask = np.zeros(gray_laplacian.shape, dtype=np.uint8)
#     _, cnts, hier = cv2.findContours(gray_laplacian.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for i in range(len(cnts)):
#         if cv2.contourArea(cnts[i]) > 10:
#             cv2.drawContours(mask, cnts, i, color=255, thickness=-1)
#     mask = cv2.dilate(mask, kernel, iterations=4)
#     out_mask = 255-cv2.bitwise_and(mask, 255-thresh)
#
#     cv2.imshow("mask", mask)
#     cv2.imshow("thresh", thresh)
#     cv2.imshow("img", img)
#     cv2.imshow("gray_laplacian", gray_laplacian)
#     cv2.imshow("out_mask", out_mask)
#     cv2.waitKey(0)


#
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
imgs = glob.glob("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/demo/*.png")
# total = collections.Counter()
total = []
for path in tqdm(imgs, leave=False, total=len(imgs)):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    cv2.imshow("img",img)
    cv2.imshow("hsv",hsv)
    cv2.imshow("s",s)
    cv2.waitKey(0)
    total += s[s!=0].flatten().tolist()
    s[s<80]=0
    s[s>80]=255
    s = np.ascontiguousarray(s, dtype=np.uint8)
    cv2.imshow("b", s)
    cv2.waitKey(0)
    # total +=
    # _, cnts, _ = cv2.findContours(s, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # for cnt in cnts:
    #     rect = cv2.boundingRect(cnt)
    #
    #     pass

plt.hist(total, 180)
plt.show()


cv2.imshow("th2", gray)
cv2.waitKey(0)
mask = gray.copy()
mask[mask == 255] = 0
mask[mask != 0] = 255
labican = cv2.Laplacian(gray, -1, ksize=3)
ret1, th1 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5),np.uint8)
mask = cv2.erode(mask, kernel, 8)
cv2.imshow("mask", mask)
cv2.imshow("labican0", labican)
_,  labican = cv2.threshold(labican, 100, 255, cv2.THRESH_BINARY)
labican[mask==0]=0
cv2.imshow("labican", labican)
kernel3 = np.ones((5, 5),np.uint8)
labican = cv2.morphologyEx(labican, cv2.MORPH_CLOSE, np.ones((11, 11),np.uint8), 4)
cv2.imshow("morphologyEx", labican)
# labican = cv2.morphologyEx(labican, cv2.MORPH_OPEN, np.ones((5, 5),np.uint8), 4)
labican = cv2.dilate(labican, np.ones((11, 11),np.uint8), 2)
cv2.imshow("th1", th1)
cv2.imshow("labican1", labican)
final = cv2.bitwise_and(255-th1, labican)
cv2.imshow("final", final)
cv2.waitKey(0)




# img2 = cv2.imread(path2)
# # img3 = cv2.imread(path3)
# h,w = img2.shape[:2]
#
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img2_gray_laplacian = cv2.Laplacian(img2_gray, -1, 5)
# # thresh = cv2.adaptiveThreshold(img2_gray, 225, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3 ,10)
# kernel = np.ones((3, 3), dtype=np.uint8)
# _, thresh = cv2.threshold(img2_gray, 100, 255, cv2.THRESH_BINARY)
# _, img2_gray_laplacian = cv2.threshold(img2_gray_laplacian, 90, 255, cv2.THRESH_BINARY)
# # img2_gray_laplacian = cv2.dilate(img2_gray_laplacian, kernel)
# img2_gray_laplacian = cv2.morphologyEx(img2_gray_laplacian, cv2.MORPH_CLOSE, kernel, 2)
#
# cv2.imshow("img2_gray_laplacian", img2_gray_laplacian)
# cv2.waitKey(0)
#
# # a = 255-img2_gray_laplacian.copy()
# mask = np.zeros(img2_gray_laplacian.shape)
# _, cnt, hier = cv2.findContours(img2_gray_laplacian.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img2_gray_laplacian = cv2.cvtColor(img2_gray_laplacian, cv2.COLOR_GRAY2BGR)
# remain = []
# for i in range(len(cnt)):
#     if cv2.contourArea(cnt[i]) > 10:
#         print(cv2.contourArea(cnt[i]))
#         remain.append(cnt[i])
#         cv2.drawContours(mask, cnt, i, color=255, thickness=-1)
#
#
# cv2.imshow("mask", mask)
# cv2.imshow("img2_gray_laplacian", img2_gray_laplacian)
# cv2.imshow("img2_gray", img2_gray)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)