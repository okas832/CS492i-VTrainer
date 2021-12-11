import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from config import cfg

class classifier(nn.Module):

    def __init__(self, joint_channels):
        super(classifier, self).__init__()
        self.joint_channels = joint_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.img_embed = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.joint_embed = nn.Sequential(
            nn.Conv2d(in_channels=self.joint_channels, out_channels=64, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self, fm, joint_f):
        
        embedded_img = self.img_embed(self.avgpool(fm)) # 32, 256, 1, 1

        if len(joint_f.shape) != 4: # 32 17
            joint_f = joint_f.unsqueeze(-1).unsqueeze(-1) # 32, 17, 1, 1

        embedded_joint = self.joint_embed(joint_f) # 32, 256, 1, 1
        
        concated_feature = torch.cat((embedded_img, embedded_joint), dim=1) # 32, 512, 1, 1
        
        out = self.layer(concated_feature)

        return out
