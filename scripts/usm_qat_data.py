import random
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def stratified_train_partition(targets, num_classes, validation_per_class, seed):
    if validation_per_class < 1:
        raise ValueError("validation_per_class must be positive")
    indices_by_class = {class_index: [] for class_index in range(num_classes)}
    for index, target in enumerate(targets):
        target = int(target)
        if target not in indices_by_class:
            raise ValueError(f"Unexpected class index in training data: {target}")
        indices_by_class[target].append(index)
    if any(len(indices) <= validation_per_class for indices in indices_by_class.values()):
        raise ValueError("Each class must have training samples beyond the validation allocation")

    rng = random.Random(seed)
    validation = []
    bn_calibration = []
    for class_index in range(num_classes):
        class_indices = indices_by_class[class_index]
        rng.shuffle(class_indices)
        validation.extend(class_indices[:validation_per_class])
        bn_calibration.extend(class_indices[validation_per_class:])
    return validation, bn_calibration


def make_train_imagefolders(dataset_root):
    train_root = Path(dataset_root) / "train"
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    evaluation_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return (
        ImageFolder(str(train_root), transform=train_transform),
        ImageFolder(str(train_root), transform=evaluation_transform),
    )


def make_test_imagefolder(dataset_root):
    test_root = Path(dataset_root) / "test"
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return ImageFolder(str(test_root), transform=transform)