from model_architectures import *

models = {
    "resnet": ECGNet_ResNet,
    "dense": ECGNet_DenseNet,
    "inception": ECGNet_Inception,
    "vgg": ECGNet_VGG,
}
