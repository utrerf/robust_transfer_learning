## TODO: 
## - mean and stdev for FMNIST
## - CIFARDS
## - STL
## - Num classes should be pulled from helpers
## ---------
from robustness import model_utils, train, defaults
from torchvision import datasets
from robustness.datasets import *

TRAIN_TRANSFORMS_DEFAULT_DOWNSCALE = lambda downscale, upscale: transforms.Compose([
            transforms.Resize(downscale),
            transforms.Resize(upscale),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT_DOWNSCALE = lambda downscale, upscale:transforms.Compose([
        transforms.Resize(downscale),
        transforms.Resize(upscale),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])


TRAIN_TRANSFORMS_MNIST = lambda size: transforms.Compose([
						transforms.Resize(size),
           				transforms.Grayscale(num_output_channels=3),
            			transforms.ToTensor(),
            			])


class STL(DataSet):
    """
    CINIC-10 dataset [DCA+18]_.
    A dataset with the same classes as CIFAR-10, but with downscaled images
    from various matching ImageNet classes added in to increase the size of
    the dataset.
    .. [DCA+18] Darlow L.N., Crowley E.J., Antoniou A., and A.J. Storkey
        (2018) CINIC-10 is not ImageNet or CIFAR-10. Report
        EDI-INF-ANC-1802 (arXiv:1810.03505)
    """
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.44671062, 0.43980984, 0.40664645]),
            'std': ch.tensor([0.26034098, 0.25657727, 0.27126738]),
            'custom_class': datasets.STL10,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(STL, self).__init__('stl', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('STL10 does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class CINIC(DataSet):
    """
    CINIC-10 dataset [DCA+18]_.
    A dataset with the same classes as CIFAR-10, but with downscaled images
    from various matching ImageNet classes added in to increase the size of
    the dataset.
    .. [DCA+18] Darlow L.N., Crowley E.J., Antoniou A., and A.J. Storkey
        (2018) CINIC-10 is not ImageNet or CIFAR-10. Report
        EDI-INF-ANC-1802 (arXiv:1810.03505)
    """
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.47889522, 0.47227842, 0.43047404]),
            'std': ch.tensor([0.24205776, 0.23828046, 0.25874835]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(CINIC, self).__init__('cinic', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CINIC does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class CUSTOM_CIFAR(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None, 
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(CUSTOM_CIFAR, self).__init__('cifar', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class FMNIST(DataSet):
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1801,0.1801,0.1801]),
            'std': ch.tensor([0.3421,0.3421,0.3421]),
            'custom_class': datasets.FashionMNIST,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_MNIST(self.size),
            'transform_test': TRAIN_TRANSFORMS_MNIST(self.size)
        }
        super(FMNIST, self).__init__('fmnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """

class KMNIST(DataSet):
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1801,0.1801,0.1801]),
            'std': ch.tensor([0.3421,0.3421,0.3421]),
            'custom_class': datasets.KMNIST,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_MNIST(self.size),
            'transform_test': TRAIN_TRANSFORMS_MNIST(self.size)
        }
        super(KMNIST, self).__init__('kmnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """

class CIFAR100(DataSet):
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR100,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(self.size),
            'transform_test': TEST_TRANSFORMS_DEFAULT(self.size)
        }
        super(CIFAR100, self).__init__('cifar100', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class SVHN(DataSet):
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4377, 0.4438,0.4728]),
            'std': ch.tensor([0.1980,0.2010,0.1970]),
            'custom_class': datasets.SVHN,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(self.size), # Transform the image
            'transform_test': TRAIN_TRANSFORMS_DEFAULT(self.size)
        }
        super(SVHN, self).__init__('SVHN', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class MNIST(DataSet):
    def __init__(self, data_path, size, **kwargs):
        self.size = size
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1307,0.1307,0.1307]),
            'std': ch.tensor([0.3081,0.3081,0.3081]),
            'custom_class': datasets.MNIST,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_MNIST(self.size), # Transform the image
            'transform_test': TRAIN_TRANSFORMS_MNIST(self.size)
        }
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return cifar_models.__dict__[arch](num_classes=self.num_classes)
