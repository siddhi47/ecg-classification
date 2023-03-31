"""
    author:siddhi47
    date: 2023-02-23
    description: This file contains the model class for the ECG classification with various pretrained networks
"""

import torch
import torch.nn.functional as F


class ECGNet_VGG(torch.nn.Module):
    """
    This class is uses VGG netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_VGG, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.6.0", "vgg16", pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.backbone = self.model

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.backbone(x)
        x = F.normalize(x)
        x = F.log_softmax(x, dim=1)
        return x


class ECGNet_ResNet(torch.nn.Module):
    """
    This class is uses ResNet netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_ResNet, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        )
        self.model.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.model(x)
        return x


class ECGNet_DenseNet(torch.nn.Module):
    """
    This class is uses DenseNet netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_DenseNet, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "densenet121", pretrained=True
        )
        self.model.classifier = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.model(x)
        return x


class ECGNet_MobileNet(torch.nn.Module):
    """
    This class is uses MobileNet netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_MobileNet, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True
        )
        self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        This function initializes the model
        """
        x = self.model(x)
        return x


class ECGNet_Inception(torch.nn.Module):
    """
    This class is uses Inception netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_Inception, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "inception_v3", pretrained=True
        )
        self.model.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.model(x)
        return x


class ECGNet_ViT(torch.nn.Module):
    """
    This class is uses VIT netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(ECGNet_ViT, self).__init__()
        self.model = torch.hub.load(
            "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True
        )
        self.model.head = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.model(x)
        return x
