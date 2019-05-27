from tensorboardX import SummaryWriter
import torch
import torchvision.transforms as T
import torchvision.models as models
import PIL.Image as Image

img = Image.open("/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/out/2019-05-15-19_26_240.png")

# transform224 = T.Compose(
#     [T.Resize((224, 224)),
#     T.ToTensor()]
# )

# imgt = transform224(img)


# imgt = torch.rand(1, 224, 224)
#
# writer = SummaryWriter(logdir='./logs', comment='image')
# writer.add_image("img", imgt)
# writer.close()


x = torch.FloatTensor([100])
y = torch.FloatTensor([500])
writer = SummaryWriter(logdir='./logs', comment='train')
for epoch in range(100):
    x /= 1.5
    y /= 1.5
    loss = y - x
    # with SummaryWriter(logdir='./logs', comment='train') as writer: #可以直接使用python的with语法，自动调用close方法
    writer.add_histogram('his/x', x, epoch)
    writer.add_histogram('his/y', y, epoch)
    writer.add_scalar('data/x', x, epoch)
    writer.add_scalar('data/y', y, epoch)
    writer.add_scalar('data/loss', loss, epoch)
    writer.add_scalars('data/data_group', {'x': x, 'y': y, 'loss': loss}, epoch)

writer.close()

# resnet18 = models.resnet18(pretrained=True)
#
# imgt = imgt.view(-1, 3, 224, 224)
#
# out = resnet18(imgt)
#
# with SummaryWriter(logdir='./logs', comment='vgg16') as writer:
#     writer.add_graph(resnet18, (imgt,))