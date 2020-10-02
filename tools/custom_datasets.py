import tools.transforms as transforms
import tools.constants as constants
import os
from robustness import imagenet_models, cifar_models
from robustness.datasets import DataSet, CIFAR
import torch as ch
from torchvision import datasets


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageNetTransfer(DataSet):
    def __init__(self, data_path, num_transf_classes=1000, **kwargs):
        self.num_classes = num_transf_classes
        imagenet_size = 224
        transform_type_to_transform = {
                'default' : (transforms.TRAIN_TRANSFORMS_DEFAULT(imagenet_size),
                             transforms.TEST_TRANSFORMS_DEFAULT(imagenet_size)),
                'black_n_white' : (transforms.BLACK_N_WHITE(imagenet_size),
                                   transforms.BLACK_N_WHITE(imagenet_size))
                }
        if kwargs['downscale']:
            transform_type_to_transform = {
                    'default' : (transforms.TRAIN_TRANSFORMS_DOWNSCALE(kwargs['downscale_size'], imagenet_size),
                                 transforms.TEST_TRANSFORMS_DOWNSCALE(kwargs['downscale_size'], imagenet_size)),
                    'black_n_white' : (transforms.BLACK_N_WHITE_DOWNSCALE(kwargs['downscale_size'], imagenet_size),
                                       transforms.BLACK_N_WHITE_DOWNSCALE(kwargs['downscale_size'], imagenet_size))
                    }
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'std': ch.tensor(kwargs['std']), 
            'custom_class': kwargs['custom_class'],
            'label_mapping': None,
            'transform_train': transform_type_to_transform[kwargs['transform_type']][0],
            'transform_test': transform_type_to_transform[kwargs['transform_type']][1]
        }
#        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        self.name = kwargs['name']
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010] 

# this class is used when we're training from scratch instead of from a pre-trained imagenet model
class CIFAR(DataSet):
    def __init__(self, num_classes, data_path=None, **kwargs):
        self.name = f'cifar{num_classes}'
        
        num_classes_to_custom_class = {
                10: datasets.CIFAR10,
                100: datasets.CIFAR100
                }

        ds_kwargs = {
            'num_classes': num_classes,
            'mean': ch.tensor(CIFAR_MEAN),
            'std': ch.tensor(CIFAR_STD),
            'custom_class': num_classes_to_custom_class[num_classes],
            'label_mapping': None,
            'transform_train': transforms.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': transforms.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CIFAR, self).__init__(f'cifar{num_classes}', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        if pretrained:
            raise ValueError('CIFAR100 does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=num_classes)

name_to_dataset = {
        'caltech101_stylized': {'num_classes':101, 'custom_class':None, 'transform_type':'default',
                       'mean':IMAGENET_MEAN, 'std':IMAGENET_STD},
        
        'food_stylized': {'num_classes':101, 'custom_class':None, 'transform_type':'default',
                 'mean':[0.5493, 0.4450, 0.3435], 'std':[0.2730, 0.2759, 0.2800]},
        
        'caltech101': {'num_classes':101, 'custom_class':None, 'transform_type':'default',
                       'mean':IMAGENET_MEAN, 'std':IMAGENET_STD},
        
        'food': {'num_classes':101, 'custom_class':None, 'transform_type':'default',
                 'mean':[0.5493, 0.4450, 0.3435], 'std':[0.2730, 0.2759, 0.2800]},

        'cifar10': {'num_classes':10, 'custom_class':datasets.CIFAR10, 'transform_type':'default',
                    'mean':CIFAR_MEAN, 'std':CIFAR_STD},
        
        'cifar100': {'num_classes':100, 'custom_class':datasets.CIFAR100, 'transform_type':'default',
                     'mean':CIFAR_MEAN, 'std':CIFAR_STD},
        
        'svhn': {'num_classes':10, 'custom_class':datasets.SVHN, 'transform_type':'default',
                 'mean':[0.4377, 0.4438,0.4728], 'std':[0.1980,0.2010,0.1970]},

        # TODO: Get mean and std for fmnist
        'fmnist': {'num_classes':10, 'custom_class':datasets.FashionMNIST, 'transform_type':'black_n_white', 
                   'mean':[0.1801,0.1801,0.1801], 'std':[0.3421,0.3421,0.3421]},
        
        'kmnist': {'num_classes':10, 'custom_class':datasets.KMNIST, 'transform_type':'black_n_white',
                   'mean':[0.1801,0.1801,0.1801], 'std':[0.3421,0.3421,0.3421]},
        
        'mnist': {'num_classes':10, 'custom_class':datasets.MNIST, 'transform_type':'black_n_white',
                  'mean':[0.1307,0.1307,0.1307], 'std':[0.3081,0.3081,0.3081]}
        }

def make_dataset(args):
    return ImageNetTransfer(name=args['target_dataset_name'],
                     data_path=f'{constants.base_data_path}{args["target_dataset_name"]}',
                     downscale=args['downscale'], downscale_size=args['downscale_size'], 
                     **name_to_dataset[args['target_dataset_name']])



