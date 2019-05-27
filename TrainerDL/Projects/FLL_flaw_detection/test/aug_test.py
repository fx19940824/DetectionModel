from torchvision.transforms import functional as TF

import matplotlib.pyplot as plt
import PIL
from imgaug import augmenters as iaa
import numpy as np
from torchvision.datasets import ImageFolder
img = TF.Image.open("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:02:36.156_7f731c0009c0_40.png")
img = TF.adjust_gamma(img, 1.2)
img = TF.affine(img, angle=-20, translate=(5,5),scale=0.9, shear=-15, fillcolor=np.random.randint(0, 255))
img = np.array(img)

plt.figure("dog")
plt.imshow(img)
plt.show()

# l = ImageFolder("/home/cobot/BM_fanhua/train")
# l.imgs *= 2
pass