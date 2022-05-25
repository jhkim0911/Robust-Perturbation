import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GoogLenet(nn.Module):
    def __init__(self, classCount):
        super(GoogLenet, self).__init__()

        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.classifier = Identity()
        self.logits = nn.Linear(1000, classCount)

    def forward(self, x):
        x = self.googlenet(x) # 32 * 1024
        logits = self.logits(x) #14

        return logits

class ResNet(nn.Module):
    def __init__(self, classCount):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = Identity()
        self.logits = nn.Linear(2048, classCount)

    def forward(self, x):
        x = self.resnet(x) # 32 * 1024
        logits = self.logits(x) #14

        return logits


class VGG(nn.Module):
    def __init__(self, classCount):
        super(VGG, self).__init__()

        self.vgg = models.vgg16_bn(pretrained=True)
        self.vgg.classifier = Identity()
        self.logits = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classCount),
        )

    def forward(self, x):
        x = self.vgg(x) # 32 * 1024
        logits = self.logits(x) #14

        return logits