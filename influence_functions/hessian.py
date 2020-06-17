from robustness import train
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet, CIFAR, CINIC
from torch import autograd
from torchvision import datasets
import torch as ch
import numpy as np
import argparse
import random
from scipy import linalg
# note: currently only supported for CIFAR10 and epsilon=[0,3]
sys.path.insert(1, '..')
from tools.helpers import flatten_grad, make_mask, set_seeds, eval_hessian
import tools.custom_datasets as custom_datasets


# PARSE INPUTS
parser = argparse.ArgumentParser(add_help=True)

eps_list = [0, 3]
parser.add_argument('-e',  required=False, default=0,
                    help='epsilon used to train the source dataset', type=int, choices=eps_list)
parser.add_argument('-n',  required=False, default=100,
                    help='number of images used to make hessian', type=int)
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
set_seeds('seed':1000000}

# MAKE THE MASK
dataset_size = len(data_set)
mask_sampler = make_mask(num_images, dataset_size)

# MAKE THE LOADER
num_workers  = 10
loader       = ch.utils.data.DataLoader(data_set, sampler=mask_sampler, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

# MAKE THE HESSIAN
for i, data in enumerate(loader):
    image, label = data
    output, final_inp = model(image.cuda())
    output = output.cpu()
    loss = criterion(output, label)
    loss_grad = autograd.grad(loss, model.model.fc.parameters(), create_graph=True)
    H_i = eval_hessian(loss_grad, model.model.fc)
    if i > 0:
        H += H_i
        if i%500 == 0:
            print(f'{i}th iteration')
    else:
        H = H_i
    ch.cuda.empty_cache()

if 'hessians' not in os.listdir(): mkdir('hessians')
np.save(f'hessians/H_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size', H)

# INVERT THE HESSIAN
H_inv = linalg.pinv2(H, rcond=1e-20)
if 'h_inverses' not in os.listdir(): mkdir('h_inverses')
np.save(f'h_inverses/H_inv_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size', H_inv)

