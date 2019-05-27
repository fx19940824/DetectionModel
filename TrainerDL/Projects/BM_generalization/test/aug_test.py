from torchvision.transforms import functional as TF

import matplotlib.pyplot as plt
import PIL
from imgaug import augmenters as iaa
import numpy as np
from torchvision.datasets import ImageFolder
img = TF.Image.open("/home/cobot/BM_fanhua/valid/ZC/2018.07.05_17:26:00.041_7fe570078ca00.png")
img = TF.adjust_gamma(img, 1.2)
img = TF.affine(img, angle=70, translate=(10,10),scale=0.8, shear=-15, fillcolor=np.random.randint(0, 255))
img = np.array(img)

plt.figure("dog")
plt.imshow(img)
plt.show()
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
seq = iaa.Sequential(
    iaa.SomeOf(3,
                 [
                     sometimes(iaa.Superpixels(p_replace=(0, 0.4), n_segments=(200, 400))),
                     iaa.OneOf([
                         iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                         iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
                         iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                     ]),
                     iaa.Add((-15, 15), per_channel=0.5),
                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                     iaa.AddToHueAndSaturation((-20, 20)),
                     iaa.Grayscale(alpha=(0.0, 0.5)),
                 ],
                 random_order=True
                 )
)

img = seq.augment_image(img)
plt.figure("dog")
plt.imshow(img)
plt.show()
# l = ImageFolder("/home/cobot/BM_fanhua/train")
# l.imgs *= 2
pass