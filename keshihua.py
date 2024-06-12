import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2


# 提取某一层网络特征图
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            x = module(x)
            print(name)
            if (self.extracted_layers is None) or (name in self.extracted_layers and 'fc' not in name):
                outputs[name] = x
        # print(outputs)
        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))  # 读入图片时将图片resize成(256,256)的
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


pic_dir = 'data/images/000013.jpg'
transform = transforms.ToTensor()
img = get_picture(pic_dir, transform)
# 插入维度
img = img.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = img.to(device)

net = models.resnet101(pretrained=True).to(device)

dst = './feautures'
therd_size = 256

myexactor = FeatureExtractor(submodule=net, extracted_layers=None)
output = myexactor(img)
# output是dict
# dict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])

for idx, val in enumerate(output.items()):
    k, v = val
    features = v[0]
    iter_range = features.shape[0]
    for i in range(iter_range):
        # plt.imshow(features.data.cpu().numpy()[i,:,:],cmap='jet')
        if 'fc' in k:  # 不可视化fc层
            continue

        feature = features.data.cpu().numpy()
        feature_img = feature[i, :, :]
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

        dst_path = os.path.join(dst, str(idx) + '-' + k)

        make_dirs(dst_path)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        if feature_img.shape[0] < therd_size:
            tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, tmp_img)

        dst_file = os.path.join(dst_path, str(i) + '.png')
        cv2.imwrite(dst_file, feature_img)