import torch
import torch.nn as nn
import pretrainedmodels as pm


class SeResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.se_resnext50_32x4d = pm.se_resnext50_32x4d()
        self.max_pool = torch.nn.MaxPool2d(7, stride=1)
        self.linear1 = torch.nn.Linear(2048, 768)
        self.last_linear = torch.nn.Linear(768, num_classes)
        self.batch_norm1 = torch.nn.BatchNorm1d(768)

    def set_gr(self, rg):
        for l in [self.se_resnext50_32x4d.layer0,
                  self.se_resnext50_32x4d.layer1,
                  self.se_resnext50_32x4d.layer2,
                  self.se_resnext50_32x4d.layer3,
                  self.se_resnext50_32x4d.layer4]:
            l.requires_grad = rg

    def forward(self, x):
        x = self.se_resnext50_32x4d.features(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = torch.nn.ReLU()(x)
        x = self.last_linear(x)
        return x


def create_model(num_classes):
    return SeResNet(num_classes)