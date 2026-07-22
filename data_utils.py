"""
data_utils.py

Split-MNIST class-incremental continual learning benchmark. MNIST's 10
digit classes are split into 5 sequential binary tasks:

    Task 1: digits 0, 1
    Task 2: digits 2, 3
    Task 3: digits 4, 5
    Task 4: digits 6, 7
    Task 5: digits 8, 9

The model is trained on these tasks one at a time with no access to
previous tasks' data during training. Forgetting is measured by
comparing each task's accuracy immediately after it is learned versus
after all 5 tasks have been trained.

Requires internet access and torchvision to download MNIST.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

NUM_STEPS = 25  # number of timesteps in the encoded spike train


def rate_encode(images, num_steps=NUM_STEPS):
    """
    Converts a static image into a Poisson-like spike train via rate
    coding: higher pixel intensity -> higher spike probability per step.

    Args:
        images: (B, 784) tensor with values in [0, 1].

    Returns:
        (T, B, 784) binary spike tensor.
    """
    images = images.unsqueeze(0).repeat(num_steps, 1, 1)
    return torch.bernoulli(images)


def get_split_mnist_tasks(data_dir="./data", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
    ])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    train_loaders, test_loaders = [], []

    for classes in task_classes:
        train_idx = [i for i, (_, y) in enumerate(train_set) if y in classes]
        test_idx = [i for i, (_, y) in enumerate(test_set) if y in classes]

        train_loaders.append(DataLoader(Subset(train_set, train_idx),
                                         batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_set, test_idx),
                                        batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders, task_classes


def collect_exemplars(loader, num_samples=20):
    """
    Collects a small number of examples from a task's train_loader to
    serve as an episodic replay memory buffer.
    """
    imgs, labels = [], []
    collected = 0
    for images, lbls in loader:
        imgs.append(images)
        labels.append(lbls)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    imgs = torch.cat(imgs, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    return imgs, labels
