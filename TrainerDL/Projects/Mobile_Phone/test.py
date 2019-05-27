import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

x = input("yes(y) or no(n)")
print(x)
image_dir = "/home/cobot/Desktop/1_right"
# image_dir = "/home/cobot/mobile_data/screw_train/label"
# cv2.namedWindow("test")

for k, name in enumerate(os.listdir(image_dir)):
    print(str(k) + ":" + name)
    image = cv2.imread(os.path.join(image_dir, name))
    # image[image > 253] = 253
    image[image <= 253] = 0

    # plt.imshow(image)
    # plt.pause(5)
    cv2.imwrite(os.path.join(image_dir, name), image)

# sigma = 35
# x, y = np.meshgrid(np.linspace(-63.5, 63.5, 128), np.linspace(-63.5, 63.5, 128))
#
# z = np.exp(-x ** 2 / sigma ** 2) * np.exp(-y ** 2 / sigma ** 2)
# z = z / np.max(z[:])
# cv2.imshow("xxx", z)
# cv2.waitKey()
#
# cv2.imwrite("weight.bmp", (z * 255).astype(np.uint8))

# kernel = np.zeros([128, 128], np.float)
# kernel[63:-63, 63:-63] = 1
# kernel = cv2.GaussianBlur(kernel, (64, 64), 20)
# # kernel = cv2.blur(kernel, [])
# # retval = cv2.getGaussianKernel((128, 128), 20)
#
# cv2.imshow("xxx", retval)
# cv2.waitKey()
