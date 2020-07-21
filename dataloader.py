import os
import pickle
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def init_data(type='cifar10', size=32, class_per_task=2):
    if type != 'cifar10':
        raise NotImplementedError
    ncla = 10

    path = os.path.join('data', 'cifar10')
    if not os.path.exists(path):
        os.makedirs(path)
    ts = transforms.Compose([
        transforms.Resize(size, size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_data = CIFAR10(path, train=True, transform=ts, download=True)
    test_data = CIFAR10(path, train=False, transform=ts, download=True)

    train_tasks, test_tasks = [], []
    belong = list(range(0, ncla))
    for i in range(0, ncla, class_per_task):
        train_tasks.append([])
        test_tasks.append([])
        for j in range(i, i + class_per_task):
            belong[j] = i // class_per_task

    for x, y in train_data:
        idex = belong[y]
        train_tasks[idex].append((x, y))
    for x, y in test_data:
        idex = belong[y]
        test_tasks[idex].append((x, y))

    path = os.path.join('data', 'cifar10', 'preprocessed')
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, ncla // class_per_task):
        path_i = os.path.join(path, '%d_%d_%d' % (size, class_per_task, i))
        with open(path_i, 'wb') as f:
            pickle.dump((train_tasks[i], test_tasks[i]), f)


def get_data_loader(num_task, batch_size=64, type='cifar10', size=32, class_per_task=2):
    if type != 'cifar10':
        raise NotImplementedError

    path = os.path.join('data', 'cifar10', 'preprocessed', '%d_%d_%d' % (size, class_per_task, num_task))
    if not os.path.exists(path):
        init_data(type=type, size=size, class_per_task=class_per_task)

    with open(path, 'rb') as f:
        train_data, test_data = pickle.load(f)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
