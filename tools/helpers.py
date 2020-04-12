import torch as ch
import torch.nn as nn
from torchvision import datasets

from robustness import defaults
from robustness.datasets import ImageNet, CIFAR, CINIC
from robustness.model_utils import make_and_restore_model

import cox
from cox.utils import Parameters

from tools.custom_datasets import CIFAR100, MNIST, SVHN
import tools.custom_datasets as custom_datasets

import argparse
import os
import random
import numpy as np
import pprint
import sys


def get_runtime_inputs():

  eps_list             = [0, 3, 4]
  unfrozen_blocks_list = [0, 1, 3]
  target_dataset_list  = ['cifar100', 'cifar10', 'svhn', 'fmnist', 'kmnist', 'mnist']

  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('-e',  required=False, default=0,         help='epsilon used to train the source dataset', type=int, choices=eps_list)
  parser.add_argument('-t',  required=False, default='cifar10', help='name of the target dataset', type=str, choices=target_dataset_list)
  parser.add_argument('-ub', required=False, default=0,         help='number of unfrozen blocks',  type=int, choices=unfrozen_blocks_list)
  parser.add_argument('-b',  required=False, default=128,       help='batch size',  type=int)
  parser.add_argument('-n',  required=False, default=-1,        help='number of images used in the target dataset', type=int)
  parser.add_argument('-w',  required=False, default=10,        help='number of workers used for parallel computations', type=int)
  parser.add_argument('-ne', required=False, default=30,        help='number of epochs', type=int)
  parser.add_argument('-li', required=False, default=5,         help='how often to log iterations', type=int)
  parser.add_argument('-s',  required=False, default=1000000,   help='random seed', type=int)


  args = parser.parse_args()

  eps_to_filename = {
    0: 'nat',
    3: 'imagenet_l2_3_0.pt',
    4: 'imagenet_linf_4.pt'
    }

  source_filename  = eps_to_filename[args.e]
  source_model_path="{}/models/{}".format(os.getcwd(), source_filename)

  var_dict = {
    'source_eps'             : args.e,
    'source_model_path'      : source_model_path, 
    'target_dataset_name'    : args.t, 
    'unfrozen_blocks'        : args.ub, 
    'batch_size'             : args.b,        
    'seed'                   : args.s,
    'num_workers'            : args.w, 
    'num_training_images'    : args.n,
    'num_epochs'             : args.ne,
    'log_iters'              : args.li
    }

  return var_dict

def set_seeds(var_dict):

  seed = var_dict['seed']
  random.seed(seed)
  np.random.seed(seed)
  ch.manual_seed(seed)


def load_model(var_dict):

  source_model_path = var_dict['source_model_path']

  if source_model_path[-3:] == 'nat':
    model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet('/tmp'), parallel=False, pytorch_pretrained=True)
  else: 
    model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet('/tmp'), resume_path=source_model_path, parallel=False)

  return model


def change_linear_layer_out_features(model, var_dict, num_in_features=2048):

  num_classes_dict = {
    'cifar100': 100,
    'cifar10' : 10,
    'cifards' : 10,
    'kmnist'  : 10,
    'fmnist'  : 10,
    'svhn'    : 10,
    'mnist'   : 10,
    'imagenet': 1000,
    'cinic'   : 10,
    'stl'     : 10
    }

  target_dataset_name = var_dict['target_dataset_name']
  num_out_features = num_classes_dict[target_dataset_name]
  model.model.fc = nn.Linear(in_features=num_in_features, out_features=num_out_features)
  return model
        

def re_init_and_freeze_blocks(model, var_dict):
  
  unfrozen_blocks_to_layer_name_list = {
    0: ['fc'],
    1: ['4.2', 'fc'],
    3: ['4.0', '4.1', '4.2', 'fc']
    }

  unfrozen_blocks = var_dict['unfrozen_blocks']
  layer_name_list = unfrozen_blocks_to_layer_name_list[unfrozen_blocks]

  for name, param in model.named_parameters():
      # if the name of the parameter is not one of the unfrozen blocks then freeze it
      if not any([layer_name in name for layer_name in layer_name_list]):
        param.requires_grad = False
      else:
        if   'fc.weight' in name: nn.init.normal_(param)
        elif 'fc.bias'   in name: nn.init.constant_(param, 0.0)
  return model


def make_train_and_test_set(var_dict):
  
  dataset_to_pytorchdataset = {
    'cifar10' : 'CIFAR10',
    'cifar100': 'CIFAR100',
    'mnist'   : 'MNIST',
    'imagenet': 'ImageNet',
    'svhn'    : 'SVHN',
    'stl'     : 'STL10',
    'fmnist'  : 'FashionMNIST',
    'kmnist'  : 'KMNIST'
    }

  img_size_dict= {
  'cifar10'  :  (32, 32),
  'cifar100' :  (32, 32),
  'imagenet' :  (224, 224),
  'mnist'    :  (32, 32),    # because at training time we augmented it
  'svhn'     :  (32, 32),
  'cinic'    :  (32, 32)
  }

  size = img_size_dict['imagenet']

  target_dataset_name = var_dict['target_dataset_name']
  pytorch_target_ds = dataset_to_pytorchdataset[target_dataset_name]
  data_path = '/tmp'

  if target_dataset_name in ['stl', 'svhn']:
    train_set = getattr(datasets, pytorch_target_ds)(root=data_path, split='train', download=True, 
                                   transform=custom_datasets.TRAIN_TRANSFORMS_DEFAULT(size))
    test_set = getattr(datasets, pytorch_target_ds)(root=data_path, split='test', download=True, 
                                  transform=custom_datasets.TEST_TRANSFORMS_DEFAULT(size))
  else:
    train_set = getattr(datasets, pytorch_target_ds)(root=data_path, train=True, download=True, 
                                   transform=custom_datasets.TRAIN_TRANSFORMS_DEFAULT(size))
    test_set = getattr(datasets, pytorch_target_ds)(root=data_path, train=False, download=True, 
                                  transform=custom_datasets.TEST_TRANSFORMS_DEFAULT(size))

  return train_set, test_set

def make_data_loaders(train_set, test_set, var_dict):
  
  num_training_images = var_dict['num_training_images']
  batch_size          = var_dict['batch_size']
  num_workers         = var_dict['num_workers']

  # make mask
  dataset_size = len(train_set)
  mask = np.ones(dataset_size)

  if num_training_images != -1:
    mask = np.zeros(dataset_size)
    shuffled_ix_list = [x for x in range(dataset_size)]
    random.shuffle(shuffled_ix_list)
    #print()
    if 'train_labels' in dir(train_set):
      remaining_classes_set = set(train_set.train_labels.tolist())
    elif 'labels' in dir(train_set):
      remaining_classes_set = set(train_set.labels)
    else:
      remaining_classes_set = set(train_set.targets)


    remaining_choices = num_training_images
    for ix in shuffled_ix_list:
      label = train_set[ix][1]
      if remaining_choices > len(remaining_classes_set) or label in remaining_classes_set:
        mask[ix] += 1
        remaining_choices -= 1
        if label in remaining_classes_set: 
          remaining_classes_set.remove(label)
      if remaining_choices <= 0: 
        break

    # check
    if sum(mask) != num_training_images:
      print(f"ERROR: the mask has {sum(mask)} images but it should have {num_training_images} images") 
    if len(remaining_classes_set) > 0:
      print(f"ERROR: we're missing classes: {remaining_classes_set}") 
  print('made the mask')
  mask = np.nonzero(mask)[0]
  mask_sampler = ch.utils.data.sampler.SubsetRandomSampler(mask)

  train_loader = ch.utils.data.DataLoader(train_set, sampler=mask_sampler, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers, pin_memory=True)
  test_loader  = ch.utils.data.DataLoader(test_set,                        batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers, pin_memory=True)
  return train_loader, test_loader


def make_out_store(var_dict):

  source_eps           = var_dict['source_eps']
  target_dataset_name  = var_dict['target_dataset_name']
  num_training_images  = var_dict['num_training_images']
  unfrozen_blocks      = var_dict['unfrozen_blocks']
  seed                 = var_dict['seed']

  out_dir = (os.getcwd()+ '/results/logs/'
                        + f'source_eps_{source_eps}_'
                        + f'target_dataset_{target_dataset_name}_'
                        + f'num_training_images_{num_training_images}_'
                        + f'unfrozen_blocks_{unfrozen_blocks}_'
                        + f'seed_{seed}')
  out_store = cox.store.Store(out_dir)
  return out_store


def make_train_args(var_dict):

  num_epochs = var_dict['num_epochs']
  step_lr = num_epochs//3

  log_iters  = var_dict['log_iters']
  
  train_kwargs = {
    'out_dir'     : "train_out",
    'adv_train'   : 0,
    'epochs'      : num_epochs,
    'step_lr'     : step_lr,
    "log_iters"   : log_iters,
    }

  train_args = Parameters(train_kwargs)
  train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, CIFAR)

  return train_args


def print_details(model, var_dict, train_args):

  for name, param in model.named_parameters():
      print("{}: {}".format(name, param.requires_grad))

  print('Input parameters: ')
  pprint.pprint(var_dict)  

  print('Transfer learning training parameters: ')
  pprint.pprint(train_args)



