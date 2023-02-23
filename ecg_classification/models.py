from egc_classification.model_architectures import *

models = {
    'resnet': ECGResNet,
    'dense': ECGDenseNet,
    'inception': ECGInception,
    'vgg': ECGVGG,
    }
