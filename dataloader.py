import pickle
from os.path import join as pjoin
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(folder_path): # Load the CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) from the designated folder
    train_data_file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_data_file_name = 'test_batch'
    batches_name = 'batches.meta'
    train_data_paths = [pjoin(folder_path, train_data_file_name) for train_data_file_name in train_data_file_names]
    test_data_path = pjoin(folder_path, test_data_file_name)
    batches_path = pjoin(folder_path, batches_name)

    train_data = []
    test_data = unpickle(test_data_path)
    for train_data_path in train_data_paths:
        train_data.append(unpickle(train_data_path))
    batches = unpickle(batches_path)
    label_names = batches[b'label_names']

    # The fields in each data batch include [b'batch_label', b'labels', b'data', b'filenames']
    # While 'batches.meta' is {b'num_cases_per_batch': 10000, b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'], b'num_vis': 3072}
    return {'train_data': train_data, 'test_data': test_data, 'label_names': label_names}


# Modified from https://github.com/omihub777/ViT-CIFAR/blob/main/utils.py. Pre-process the training and testing data.
def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args["size"], padding=args["padding"])
    ]
    train_transform += [transforms.RandomHorizontalFlip()]
    train_transform.append(CIFAR10Policy()) # Auto data augmentation

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args["mean"], std=args["std"])
    ]
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args["mean"], std=args["std"])
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


# Modified from https://github.com/omihub777/ViT-CIFAR/blob/main/utils.py. Only get the CIFAR-10 dataset.
def get_dataset():
    root = "data"
    transform_args = {}
    transform_args["in_c"] = 3
    transform_args["num_classes"] = 10
    transform_args["size"] = 32
    transform_args["padding"] = 4
    transform_args["mean"] = [0.4914, 0.4822, 0.4465]
    transform_args["std"] = [0.2470, 0.2435, 0.2616]
    train_transform, test_transform = get_transform(transform_args)
    train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)
    
    return train_ds, test_ds


if __name__ == '__main__':
    data = load_data(folder_path='./cifar-10')