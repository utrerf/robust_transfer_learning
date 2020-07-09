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
import os
# note: currently only supported for CIFAR10 and epsilon=[0,3]
import sys
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


H = np.load(f'hessians/H_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size.npy')

# INVERT THE HESSIAN
H_inv = linalg.pinv2(H, rcond=1e-20)
if 'h_inverses' not in os.listdir(): os.mkdir('h_inverses')
np.save(f'h_inverses/H_inv_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size', H_inv)

