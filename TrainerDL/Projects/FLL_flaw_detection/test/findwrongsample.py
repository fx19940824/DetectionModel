import os
from Projects.FLL_flaw_detection.classification.train import Generalization_CLS
import glob
from PIL import Image
from torchvision import transforms
import torch
# model = Generalization_CLS("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/classification/cfgs/train_fll_cls.cfg", is_Train=False).model
# model.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/classification/weights/model_epoch_7.pth")['net'])
#
# imgs = glob.glob("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/*/*.png")
#
# transform = transforms.Compose([
#  transforms.Resize(299),
#  transforms.CenterCrop(299),
#  transforms.ToTensor(),
#  transforms.Lambda(lambda x:x.unsqueeze(0)),
#     ])
# model.eval()

# for imgpath in imgs:
#     cls = 0 if 'bad' in imgpath else 1
#     img = Image.open(imgpath)
#     img = transform(img).cuda()
#     pred = model(img)
#     out = int(torch.max(pred, 1)[1].cpu().numpy())
#     if cls != out:
#  print(cls, out, pred, imgpath)
#

imglist = [
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:08:54.426_7f731c0009c0_19.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:09:31.332_7f731c0009c0_46.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:49:54.603_7f731c0009c0_23.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:50:43.920_7f731c0009c0_5.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_10:52:23.211_7f731c0009c0_3.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_11:07:42.174_7f731c0009c0_37.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_11:09:06.269_7f731c0009c0_11.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_11:09:26.237_7f731c0009c0_16.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_14:31:11.121_7f57c40009c0_8.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_14:32:55.397_7f57c40009c0_7.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_14:45:07.170_7f57c40009c0_26.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_14:45:36.232_7f57c40009c0_13.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_14:46:03.684_7f57c40009c0_11.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_15:45:28.742_7f57c40009c0_26.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_16:00:01.516_7f57c40009c0_34.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_16:00:01.516_7f57c40009c0_35.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_16_12_33.130_7f57c40009c0_35.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_17_19_06.834_7f57c40009c0_6.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.27_17_33_00.196_7f57c40009c0_46.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.28_17:55:16.509_7f378c0009c0_3.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.28_20:49:12.859_7fac680009c0_27.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.02.28_21:15:01.190_7fac680009c0_31.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_11:57:56.843_7f9d500009c0_31.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_11:58:20.988_7f9d500009c0_37.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_17:46:39.147_7f9d500009c0_45.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_22:20:32.959_7f9d500009c0_26.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_22:21:18.897_7f9d500009c0_39.png',
'/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/test/bad/2019.03.01_22:25:01.816_7f9d500009c0_17.png']
#
for img in imglist:
    os.system("cp %s %s" % (img, os.path.join("/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_cls/misclassify/负样本漏检", img.split('/')[-1])))
