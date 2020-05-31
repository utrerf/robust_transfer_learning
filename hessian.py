import tools.helpers as helpers
from robustness import train
import cox.store
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet, CIFAR, CINIC
from torch import autograd
from torchvision import datasets
from tools.custom_datasets import CIFAR100, MNIST, SVHN
import tools.custom_datasets as custom_datasets
import torch as ch
import numpy as np
from copy import deepcopy
import argparse
import random
from scipy import linalg

# note: currently only supported for CIFAR10 and epsilon=[0,3]

# PARSE INPUTS
parser = argparse.ArgumentParser(add_help=True)

eps_list = [0, 3]
parser.add_argument('-e',  required=False, default=0,
                    help='epsilon used to train the source dataset', type=int, choices=eps_list)
parser.add_argument('-n',  required=False, default=100,
                    help='number of images used to make hessian', type=int)
# parser.add_argument('-i',  required=False, default=0,
#                     help='starting index on training dataset', type=int)
parser.add_argument('-b',  required=False, default=50,
                    help='batch_size', type=int)
parser.add_argument('-s',  required=False, default=1000000,
                    help='seed', type=int)
parser.add_argument('-ub',  required=False, default=1000000,
                    help='number of unfrozen blocks', type=int)

args = parser.parse_args()

eps        = args.e
num_images = args.n
batch_size = args.b
seed       = args.s
ub         = args.ub
# start_idx  = args.i 

# LOAD MODEL
eps_to_model = {3: f'l2_{eps}_imagenet_to_cifar10_{ub}_ub_{num_images}_images.pt',
                0: f'nat_imagenet_to_cifar10_{ub}_ub_{num_images}_images.pt'}

source_model_path = 'models/' + eps_to_model[eps]

model, _ = make_and_restore_model(arch='resnet50_m', dataset=ImageNet('/tmp'), 
                                  resume_path=source_model_path, parallel=False)
model.eval()
criterion = ch.nn.CrossEntropyLoss()

# MAKE DATASET
size = (224, 224)
data_set = datasets.CIFAR10(root='/tmp', train=True, download=True,
                            transform=custom_datasets.TEST_TRANSFORMS_DEFAULT(size))


# SET THE SEEDS
seed = 1000000
random.seed(seed)
np.random.seed(seed)
ch.manual_seed(seed)

# MAKE THE MASK
dataset_size = len(data_set)
mask = np.ones(dataset_size)

if num_images != -1:
    mask = np.zeros(dataset_size)
    shuffled_ix_list = [x for x in range(dataset_size)]
    random.shuffle(shuffled_ix_list)

    if 'train_labels' in dir(data_set):
      remaining_classes_set = set(data_set.train_labels.tolist())
    elif 'labels' in dir(data_set):
      remaining_classes_set = set(data_set.labels)
    else:
      remaining_classes_set = set(data_set.targets)

    remaining_choices = num_images
    for ix in shuffled_ix_list:
      label = data_set[ix][1]
      if remaining_choices > len(remaining_classes_set) or label in remaining_classes_set:
        mask[ix] += 1
        remaining_choices -= 1
        if label in remaining_classes_set: 
          remaining_classes_set.remove(label)
      if remaining_choices <= 0: 
        break

print('made the mask')
mask = np.nonzero(mask)[0]
mask_sampler = ch.utils.data.sampler.SubsetRandomSampler(mask)

# MAKE THE LOADER
num_workers         = 10
loader = ch.utils.data.DataLoader(data_set, sampler=mask_sampler, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

# MAKE THE HESSIAN
for i, data in enumerate(loader):
    image, label = data
    output, final_inp = model(image.cuda())
    output = output.cpu()
    loss = criterion(output, label)
    loss_grad = autograd.grad(loss, model.model.fc.parameters(), create_graph=True)
    H_i = helpers.eval_hessian(loss_grad, model.model.fc)
    if i > 0:
        H += H_i
    else:
        H = H_i
    ch.cuda.empty_cache()
    
np.save(f'agg_hessians/H_{ub}_ub_{eps}_eps_{num_images}_images_{b}_batch_size', h)

# INVERT THE HESSIAN
H_inv = linalg.pinv2(H, rcond=1e-20)
np.save(f'h_inverses/H_inv_{ub}_ub_{eps}_eps_{num_images}_images_{b}_batch_size', h)

# MAKE THE GRADIENTS
batch_size          = 1
num_workers         = 1
loader = ch.utils.data.DataLoader(data_set, sampler=mask_sampler, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

flag = 0
# get influence
for i, data in enumerate(loader):
    image, label = data
    output, final_inp = model(image.cuda()) 
    loss = criterion(output.cpu(), label)
    loss_grad = autograd.grad(loss, model.model.fc.parameters(), create_graph=True)
    loss_grad = flatten_grad(loss_grad)    
    ch.cuda.empty_cache()
    if flag == 0:
        grad = loss_grad
        flag = 1
    else:
        grad = np.vstack((grad, loss_grad))
        if i%100 == 0:
            np.save('train_grad/train_grad_{eps}_eps_{i}_idx_{ub}_ub_{num_images}_images', np.array(grad))
            flag = 0

np.save('train_grad/train_grad_{eps}_eps_{len(loader)}_idx_{ub}_ub_{num_images}_images', np.array(grad))
