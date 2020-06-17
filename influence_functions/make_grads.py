import os
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet, CIFAR, CINIC
from torch import autograd
from torchvision import datasets
import torch as ch
import numpy as np
import argparse
# note: currently only supported for CIFAR10 and epsilon=[0,3]
import sys
sys.path.insert(1, '..')
from tools.helpers import flatten_grad
import tools.custom_datasets as custom_datasets


# PARSE INPUTS
parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('-e',  required=False, default=0,
                    help='epsilon used to train the source dataset', type=int, choices=[0, 3])
parser.add_argument('-n',  required=False, default=3200,
                    help='number of images used to make hessian', type=int)
parser.add_argument('-ub',  required=False, default=3,
                    help='number of unfrozen blocks', type=int)
parser.add_argument('-t',  required=False, default='train',
                    help='train or test?', type=str, choices=['train', 'test'])

args = parser.parse_args()

eps        = args.e
num_images = args.n
ub         = args.ub
data_type  = args.t
isTrain = (data_type == 'train')

# LOAD MODEL
eps_to_model = {3: f'l2_{eps}_imagenet_to_cifar10_{ub}_ub_{num_images}_images.pt',
                0: f'nat_imagenet_to_cifar10_{ub}_ub_{num_images}_images.pt'}

source_model_path = '../models/' + eps_to_model[eps]

model, _ = make_and_restore_model(arch='resnet50_m', dataset=ImageNet('/tmp'), 
                                  resume_path=source_model_path, parallel=False)
model.eval()
criterion = ch.nn.CrossEntropyLoss()

# MAKE DATASET
size = (224, 224)
data_set = datasets.CIFAR10(root='/tmp', train=isTrain, download=True,
                            transform=custom_datasets.TEST_TRANSFORMS_DEFAULT(size))

# MAKE THE GRADIENTS
batch_size          = 1
num_workers         = 1
loader = ch.utils.data.DataLoader(data_set, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

import os
base1 = f'{data_type}_grad'
base2 = f'{eps}_eps_{ub}_ub_{num_images}_images'
if base1 not in os.listdir():
    os.mkdir(base1)
if base2 not in os.listdir(base1):
    os.mkdir(base1 + '/' + base2)

# get influence
for i, data in enumerate(loader):
    image, label = data
    output, final_inp = model(image.cuda()) 
    loss = criterion(output.cpu(), label)
    loss_grad = autograd.grad(loss, model.model.fc.parameters(), create_graph=True)
    loss_grad = flatten_grad(loss_grad)    
    ch.cuda.empty_cache()
    if i%1000 == 0:
        grad = loss_grad
    else:
        grad = np.vstack((grad, loss_grad))
        if ((i+1)%1000) == 0:
            np.save(base1+'/'+base2+'/' + f'{i}_end_idx', np.array(grad))

