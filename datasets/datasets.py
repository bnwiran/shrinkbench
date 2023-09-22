import os

from . import presets
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode

import time

from . import places365

_constructors = {
    'MNIST': datasets.MNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
    'ImageNet': datasets.ImageNet,
    'Places365': places365.Places365
}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    if path is None:
        # Look for the dataset in known paths
        if 'DATAPATH' in os.environ:
            path = os.environ['DATAPATH'] + '/' + dataset
        else:
            raise ValueError(f"No path specified for dataset {dataset}. A path must be provided, \n \
                           or the folder must be listed in your DATAPATH")

    return path


def dataset_builder(dataset, train=True, normalize=None, preproc=None, path=None):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
    """
    if preproc is not None:
        preproc += [transforms.ToTensor()]
        if normalize is not None:
            preproc += [normalize]
        preproc = transforms.Compose(preproc)

    kwargs = {'transform': preproc,
              'download': True}
    if dataset == 'ImageNet':
        kwargs['split'] = 'train' if train else 'val'
        del kwargs['download']
    else:
        kwargs['train'] = train

    path = dataset_path(dataset, path)

    return _constructors[dataset](path, **kwargs)


def load_data(root_dir, dataset_name, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args['val_resize_size'],
        args['val_crop_size'],
        args['train_crop_size'],
    )
    interpolation = InterpolationMode(args['interpolation'])

    print("Loading training data")
    st = time.time()

    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    auto_augment_policy = args.get("auto_augment", None)
    random_erase_prob = args.get("random_erase", 0.0)
    ra_magnitude = args.get("ra_magnitude", None)
    augmix_severity = args.get("augmix_severity", None)
    dataset = _constructors[dataset_name](
        root=root_dir,
        transform=presets.ClassificationPresetTrain(
            mean=args['mean'],
            std=args['std'],
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=args['backend'],
            use_v2=args['use_v2'],
        ),
        train=True
    )

    print("Took", time.time() - st)

    print("Loading validation data")
    if args['weights'] and args['test_only']:
        weights = torchvision.models.get_weight(args['weights'])
        preprocessing = weights.transforms(antialias=True)
        if args['backend'] == "tensor":
            preprocessing = transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            backend=args['backend'],
            use_v2=args['use_v2'],
        )

    dataset_test = _constructors[dataset_name](
        root_dir,
        preprocessing,
        train=False
    )

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def MNIST(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder('MNIST', train, normalize, [], path)
    dataset.shape = (1, 28, 28)
    dataset.val_size = 0.2
    return dataset


def CIFAR10(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('CIFAR10', train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    dataset.val_size = 0.2
    return dataset


def CIFAR100(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR100
    """
    dataset_name = 'CIFAR100'
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    root_dir = dataset_path(dataset_name, path)
    args = {'val_resize_size': 256, 'val_crop_size': 224, 'train_crop_size': 224, 'interpolation': 'bilinear',
            'backend': 'PIL', 'use_v2': False, 'auto_augment': 'imagenet', 'random_erase': 0.2, 'weights': None,
            'test_only': False, 'mean': mean, 'std': std}
    return load_data(root_dir, dataset_name, args)


def ImageNet(train=True, path=None):
    """Thin wrapper around torchvision.datasets.ImageNet
    """
    # ImageNet loading from files can produce benign EXIF errors
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('ImageNet', train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    dataset.val_size = 0.2
    return dataset


def Places365(train=True, path=None):
    """Thin wrapper around .datasets.places365.Places365
    """

    # Note : Bolei used the normalization for Imagenet, not the one for Places!
    # # https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
    # So these are kept so weights are compatible
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize = transforms.Normalize((mean,), (std,))
    if train:
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('Places365', train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    dataset.val_size = 0.2
    return dataset
