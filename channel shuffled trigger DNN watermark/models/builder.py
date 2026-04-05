from models.alexnet import AlexNet
from models.efficientnet import efficientnet_b0
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.wide_resnet import Wide_ResNet


def build_model(dataset):
    if dataset == "MNIST":
        return resnet18(num_classes=10, penultimate_2d=True)
    if dataset == "CIFAR10":
        return resnet18(num_classes=10, penultimate_2d=False)
    if dataset == "CIFAR100":
        return Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0, num_classes=100)
    if dataset == "FOOD101":
        return efficientnet_b0(num_classes=101)
    if dataset == "Caltech101":
        return AlexNet(101)
    if dataset == "CALTECH256":
        return MobileNetV2(num_classes=256)
    raise ValueError(f"Unsupported dataset: {dataset}")
