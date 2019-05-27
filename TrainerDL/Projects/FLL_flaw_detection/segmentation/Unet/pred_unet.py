import torch
from Projects.FLL_flaw_detection.segmentation.Unet.unet.unet_model import UNet
from PIL import Image
import numpy as np
import cv2
import time
import glob
from torchsummary import summary
def predict_unet(net, img, threshold=0.5):
    shape = img.size
    img = np.array(img.resize((224, 224)), dtype=np.float32).transpose([2, 0, 1])[np.newaxis,...]
    img /= 255.0
    img = torch.from_numpy(img).cuda()
    st = time.time()
    out = net(img)
    torch.cuda.synchronize()
    t = time.time()-st
    out = out.data.cpu().numpy().squeeze()
    out = out > threshold
    out = cv2.resize((out * 255).astype(np.uint8), shape)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out, t


if __name__ == '__main__':
    net = UNet(3, 1).cuda().eval()
    summary(net, (3, 224, 224), (1))
    total_time = 0
    net.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/weights/CP12.pth"))
    imgs = glob.glob("/home/cobot/FLL_black_seg/image/test/*.png")
    for path in imgs:

        img = Image.open(path)
        mask, t = predict_unet(net, img)
        total_time += t
        cv2.imwrite(path.replace('.png', '_mask.png'), mask)
    print(total_time/len(imgs))
    pass