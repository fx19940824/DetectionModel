import torch
from Unet.unet.unet_model import UNet
from PIL import Image
import numpy as np
import cv2
import time
import glob
def predict_unet(net, img, threshold=0.5):
    shape = img.size
    img = np.array(img.resize((500, 500)), dtype=np.float32).transpose([2, 0, 1])[np.newaxis,...]
    img /= 255.0
    img = torch.from_numpy(img).cuda()
    out = net(img)
    torch.cuda.synchronize()
    out = out.data.cpu().numpy().squeeze()
    out = out > threshold
    out = cv2.resize((out * 255).astype(np.uint8), shape)
    return out


if __name__ == '__main__':
    imgs = glob.glob("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/all/*.png")
    net = UNet(3, 1).cuda().eval()
    net.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/weights/Unet_best500.pth"))
    for path in imgs:

        img = Image.open(path)
        mask = predict_unet(net, img)
        cv2.imwrite(path.replace(".png", "_mask.png"), mask)
    # cv2.imshow("img", np.array(img, dtype=np.uint8))
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    pass