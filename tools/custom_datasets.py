import tools.transforms as transforms
import tools.constants as constants
import os
from robustness import imagenet_models, cifar_models
from robustness.datasets import DataSet, CIFAR
import torch as ch
from torchvision import datasets

class FOOD(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'food'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 101,
            'mean': ch.tensor([0.54930437, 0.44500041, 0.34350203]),
            'std': ch.tensor([0.272926  , 0.27589517, 0.27998645]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(size),
            'transform_test': transforms.TEST_TRANSFORMS_DEFAULT(size)
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(FOOD, self).__init__('food', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, 
                                        pretrained=pretrained)

class CIFAR10_Transfered(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'cifar10'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None, 
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(size),
            'transform_test': transforms.TEST_TRANSFORMS_DEFAULT(size)
        }
        super(CIFAR10_Transfered, self).__init__('cifar10_transf', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        # pretrained on Imagenet
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


class FMNIST(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'fmnist'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1801,0.1801,0.1801]),
            'std': ch.tensor([0.3421,0.3421,0.3421]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_MNIST(size),
            'transform_test': transforms.TRAIN_TRANSFORMS_MNIST(size)
        }
        super(FMNIST, self).__init__('fmnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


class KMNIST(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'kmnist'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1801,0.1801,0.1801]),
            'std': ch.tensor([0.3421,0.3421,0.3421]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_MNIST(size),
            'transform_test': transforms.TRAIN_TRANSFORMS_MNIST(size)
        }
        super(KMNIST, self).__init__('kmnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


class CIFAR100_Transfered(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'cifar100'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        self.size = size
        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR100,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': transforms.TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(CIFAR100, self).__init__('cifar100_transf', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


class CIFAR100(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'cifar100'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        self.size = size
        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': transforms.TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(CIFAR100, self).__init__('cifar100', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        if pretrained:
            raise ValueError('CIFAR100 does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=100)


class SVHN(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.name = 'svhn'
        if data_path == None: 
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4377, 0.4438,0.4728]),
            'std': ch.tensor([0.1980,0.2010,0.1970]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(size), 
            'transform_test': transforms.TRAIN_TRANSFORMS_DEFAULT(size)
        }
        super(SVHN, self).__init__('SVHN', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)

class MNIST(DataSet):
    def __init__(self, data_path=None, size=224, **kwargs):
        self.size = size
        if data_path == None:
            data_path = os.path.abspath(f'{constants.data_path}/{self.name}')
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1307,0.1307,0.1307]),
            'std': ch.tensor([0.3081,0.3081,0.3081]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': transforms.TRANSFORMS_MNIST(self.size), 
            'transform_test': transforms.TRANSFORMS_MNIST(self.size)
        }
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


name_to_dataset_class = {
        'food':     FOOD,
        'cifar10':  CIFAR10_Transfered,
        'cifar100': CIFAR100_Transfered,
        'kmnist':   KMNIST,
        'mnist':    MNIST,
        'fmnist':   FMNIST,
        'svhn':     SVHN
        }

name_to_from_scratch_dataset_class = name_to_dataset_class.copy()
name_to_from_scratch_dataset_class['cifar10'] = CIFAR
name_to_from_scratch_dataset_class['cifar100'] = CIFAR100
