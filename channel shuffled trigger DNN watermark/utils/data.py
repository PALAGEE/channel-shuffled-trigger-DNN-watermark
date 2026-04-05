import os

import torchvision
from torch.utils.data import DataLoader


def _num_workers():
    return 0 if os.name == "nt" else 4


def _convert_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _require_split_folders(root):
    train_path = os.path.join(root, "train")
    test_path = os.path.join(root, "test")
    if not os.path.isdir(train_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(
            f"Expected dataset split folders at '{train_path}' and '{test_path}'."
        )
    return train_path, test_path


def train_test_loader(run_config):
    dataset = run_config["dataset"]
    batch_size = run_config["batch_size"]
    data_root = run_config["dataset_path"]
    workers = _num_workers()

    if dataset == "MNIST":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
    elif dataset == "CIFAR10":
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)
    elif dataset == "CIFAR100":
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        trainset = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=test_transform)
    elif dataset in {"FOOD101", "Caltech101", "CALTECH256"}:
        root = os.path.join(data_root, run_config["data_subdir"])
        train_path, test_path = _require_split_folders(root)
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(_convert_to_rgb),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(_convert_to_rgb),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        trainset = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
        testset = torchvision.datasets.ImageFolder(test_path, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return trainset, testset, trainloader, testloader
