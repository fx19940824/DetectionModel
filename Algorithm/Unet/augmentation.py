from imgaug import augmenters as iaa
seq = iaa.Sequential(
        iaa.SomeOf(3,
                   [
                       iaa.OneOf([
                         iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                         iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
                         iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                       iaa.Add((-10, 10), per_channel=0.5),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                       iaa.AddToHueAndSaturation((-20, 20)),
                       iaa.Grayscale(alpha=(0.0, 0.5)),

                   ],
                   random_order=True
                   )
    )


def augmentation(img):
    img = seq.augment_image(img)
    return img