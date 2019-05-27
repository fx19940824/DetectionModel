import torch
from torch import nn
from classifications.utils import model_factory
from torchsummary import summary


class DAECNN(nn.Module):
    def __init__(self, modelname='bninception', n_class=2, feature_extract=False, use_pretrained=True):
        super(DAECNN, self).__init__()
        self.DAE = DAE()
        self.CNN = model_factory.initialize_model(modelname, n_class, feature_extract, use_pretrained)

    def forward(self, input):
        recon_image = self.DAE(input)
        preds = self.CNN(recon_image)
        return recon_image, preds


class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encode_block_1 = encode_block(3)
        self.encode_block_1_downsample = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.encode_block_2 = encode_block()
        self.encode_block_2_downsample = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.encode_block_3 = encode_block()

        self.decode_block_2_upsample = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.decode_block_2 = decode_block()
        self.decode_block_1_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.decode_block_1 = decode_block(in_c=384)

        self.decode = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding=0, stride=1)

    def forward(self, input):
        feat_encode_1 = self.encode_block_1(input)
        feat_encode_1_downsample = self.encode_block_1_downsample(feat_encode_1)

        feat_encode_2 = self.encode_block_2(feat_encode_1_downsample)
        feat_encode_2_downsample = self.encode_block_2_downsample(feat_encode_2)

        feat_encode_3 = self.encode_block_3(feat_encode_2_downsample)

        feat_decode_2_upsample = self.decode_block_2_upsample(feat_encode_3)
        feat_decode_2_upsample = torch.cat((feat_decode_2_upsample, feat_encode_2), 1)
        feat_decode_2 = self.decode_block_2(feat_decode_2_upsample)

        feat_decode_1_upsample = self.decode_block_1_upsample(feat_decode_2)
        feat_decode_1_upsample = torch.cat((feat_decode_1_upsample, feat_encode_1), 1)
        feat_decode_1 = self.decode_block_1(feat_decode_1_upsample)

        feat_decode = self.decode(feat_decode_1)
        output = input + feat_decode
        return output


class encode_block(nn.Module):
    def __init__(self, in_c=128):
        super(encode_block, self).__init__()
        self.encoder_1_conv = nn.Conv2d(in_channels=in_c, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.encoder_1_bn = nn.BatchNorm2d(128, affine=True)
        self.encoder_1_relu = nn.ReLU()
        self.encoder_2_conv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, padding=0, stride=1)
        self.encoder_2_bn = nn.BatchNorm2d(32, affine=True)
        self.encoder_2_relu = nn.ReLU()
        self.encoder_3_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.encoder_3_bn = nn.BatchNorm2d(32, affine=True)
        self.encoder_3_relu = nn.ReLU()
        self.encoder_4_conv = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.encoder_4_bn = nn.BatchNorm2d(128, affine=True)
        self.encoder_4_relu = nn.ReLU()

    def forward(self, input):
        x = self.encoder_1_conv(input)
        x = self.encoder_1_bn(x)
        x = self.encoder_1_relu(x)

        res_x = self.encoder_2_conv(x)
        res_x = self.encoder_2_bn(res_x)
        res_x = self.encoder_2_relu(res_x)
        res_x = self.encoder_3_conv(res_x)
        res_x = self.encoder_3_bn(res_x)
        res_x = self.encoder_3_relu(res_x)
        res_x = self.encoder_4_conv(res_x)
        res_x = self.encoder_4_bn(res_x)
        res_x = self.encoder_4_relu(res_x)

        x = x+res_x

        return x


class decode_block(nn.Module):
    def __init__(self, in_c=256):
        super(decode_block, self).__init__()
        self.decoder_1_conv = nn.Conv2d(in_channels=in_c, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.decoder_1_bn = nn.BatchNorm2d(256, affine=True)
        self.decoder_1_relu = nn.ReLU()
        self.decoder_2_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.decoder_2_bn = nn.BatchNorm2d(64, affine=True)
        self.decoder_2_relu = nn.ReLU()
        self.decoder_3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.decoder_3_bn = nn.BatchNorm2d(64, affine=True)
        self.decoder_3_relu = nn.ReLU()
        self.decoder_4_conv = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.decoder_4_bn = nn.BatchNorm2d(256, affine=True)
        self.decoder_4_relu = nn.ReLU()

    def forward(self, input):
        x = self.decoder_1_conv(input)
        x = self.decoder_1_bn(x)
        x = self.decoder_1_relu(x)

        res_x = self.decoder_2_conv(x)
        res_x = self.decoder_2_bn(res_x)
        res_x = self.decoder_2_relu(res_x)
        res_x = self.decoder_3_conv(res_x)
        res_x = self.decoder_3_bn(res_x)
        res_x = self.decoder_3_relu(res_x)
        res_x = self.decoder_4_conv(res_x)
        res_x = self.decoder_4_bn(res_x)
        res_x = self.decoder_4_relu(res_x)

        x = x+res_x

        return x


daecnn = DAE().cuda()
summary(daecnn, (3, 224, 224), (1))