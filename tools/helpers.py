import torch as ch
from torch import autograd
import torch.nn as nn
from torchvision import datasets

from robustness import defaults
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model

import cox
from cox.utils import Parameters

import tools.custom_datasets as custom_datasets
import tools.constants as constants

import argparse
import os
import random
import numpy as np
import pprint
import sys


def get_runtime_inputs():

  eps_list             = list(constants.eps_to_filename.keys())
  unfrozen_blocks_list = [-1, 0, 1, 3, 9]
  #                         'food101', 'aircraft', 'caltech101', 'pets', 'cars', 'dtd']
  target_dataset_list = list(custom_datasets.name_to_dataset.keys())

  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('-e',  required=False, default=0,         help='epsilon used to train the source dataset. -1 means train from scratch', type=float, choices=eps_list)
  parser.add_argument('-t',  required=False, default='cifar10', help='name of the target dataset', type=str, choices=target_dataset_list)
  parser.add_argument('-ub', required=False, default=0,         help='number of unfrozen blocks. -1 means all blocks unfrozen',  type=int, choices=unfrozen_blocks_list)
  parser.add_argument('-b',  required=False, default=128,       help='batch size',  type=int)
  parser.add_argument('-n',  required=False, default=-1,        help='number of images used in the target dataset', type=int)
  parser.add_argument('-w',  required=False, default=10,        help='number of workers used for parallel computations', type=int)
  parser.add_argument('-ne', required=False, default=30,        help='number of epochs', type=int)
  parser.add_argument('-li', required=False, default=5,         help='how often to log iterations', type=int)
  parser.add_argument('-s',  required=False, default=1000000,   help='random seed', type=int)
  parser.add_argument('-d',  required=False, default=False,     help='downscale to lower res?', type=bool)
  parser.add_argument('-ds', required=False, default=32,        help='downscaled resolution',   type=int)
  parser.add_argument('-lr', required=False, default=0.1,       help='learning rate',   type=float)
 # parser.add_argument('-lp', required=False, default=False,     help='use the 32x32 low pass?',   type=bool)

  args = parser.parse_args()

  var_dict = {
    'source_eps'             : args.e,
    'target_dataset_name'    : args.t, 
    'unfrozen_blocks'        : args.ub, 
    'batch_size'             : args.b,        
    'seed'                   : args.s,
    'num_workers'            : args.w, 
    'num_training_images'    : args.n,
    'num_epochs'             : args.ne,
    'log_iters'              : args.li,
    'downscale'              : args.d,
    'downscale_size'         : args.ds,
    'learning_rate'          : args.lr,
    # 'low_pass'               : args.lp
    }

  return var_dict


def set_seeds(var_dict):
  seed = var_dict['seed']
  random.seed(seed)
  np.random.seed(seed)
  ch.manual_seed(seed)


def change_linear_layer_out_features(model, var_dict, dataset, num_in_features=2048):
  num_out_features = dataset.num_classes
  model.model.fc = nn.Linear(in_features=num_in_features, out_features=num_out_features)
  return model
  

def load_model(var_dict, is_Transfer, pretrained, dataset):
  if is_Transfer:
    if var_dict['source_eps'] > 0:
      resume_path = os.path.abspath('models/'+constants.eps_to_filename[var_dict['source_eps']])
      model, _ = make_and_restore_model(arch='resnet50', dataset=dataset, parallel=False, resume_path=resume_path, pytorch_pretrained=False)
    else: 
      model, _ = make_and_restore_model(arch='resnet50', dataset=dataset, parallel=False, resume_path=None, pytorch_pretrained=True)
  else:
    model = dataset.get_model(arch='resnet50', pretrained=False)

  return model


def re_init_and_freeze_blocks(model, var_dict):

  unfrozen_blocks_to_layer_name_list = {
    0: ['fc'],
    1: ['fc', '4.2'],
    3: ['fc', '4.2', '4.1', '4.0'],
    9: ['fc', '4.2', '4.1', '4.0', '3.5', '3.4', '3.3', '3.2', '3.1', '3.0']
    }
  
  layer_name_list = unfrozen_blocks_to_layer_name_list[var_dict['unfrozen_blocks']]

  for name, param in model.named_parameters():
      # if the name of the parameter is not one of the unfrozen blocks then freeze it
      if not any([layer_name in name for layer_name in layer_name_list]):
        param.requires_grad = False
      else:
        if   'fc.weight' in name: nn.init.normal_(param)
        elif 'fc.bias'   in name: nn.init.constant_(param, 0.0)
  return model

def make_out_store(var_dict):

  out_dir = (os.getcwd()+ '/results/logs/'
                        + f'source_eps_{var_dict["source_eps"]}_'
                        + f'target_dataset_{var_dict["target_dataset_name"]}_'
                        + f'num_training_images_{var_dict["num_training_images"]}_'
                        + f'unfrozen_blocks_{var_dict["unfrozen_blocks"]}_'
                        + f'seed_{var_dict["seed"]}_'
                        + f'downscaled_{var_dict["downscale"]}_')
  out_store = cox.store.Store(out_dir)
  return out_store


def make_train_args(var_dict):
  
  train_kwargs = {
    'out_dir'     : "train_out",
    'adv_train'   : 0,
    'epochs'      : var_dict['num_epochs'],
    'step_lr'     : var_dict['num_epochs']//3,
    'log_iters'   : var_dict['log_iters'],
    'learning_rate': var_dict['learning_rate']
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


def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else ch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = ch.zeros(l, l).cpu().double()
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], model.parameters(), retain_graph=True)
        grad2rd = (grad2rd[0].cpu().double(), grad2rd[1].cpu().double())
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else ch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().double().data.numpy()


def flatten_grad(grad):
    cnt = 0
    for g in grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else ch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    return g_vector.cpu().double().data.numpy()

def get_runtime_inputs_for_influence_functions():
    # PARSE INPUTS
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-e',  required=False, default=0,
                        help='epsilon used to train the source dataset', type=int, choices=[0, 3])
    parser.add_argument('-n',  required=False, default=3200,
                        help='number of images used to make hessian', type=int)
    parser.add_argument('-ub',  required=False, default=3,
                        help='number of unfrozen blocks', type=int)
    parser.add_argument('-s',  required=False, default=1000000,
                        help='seed', type=int)
    ds_list = ['cifar10', 'svhn']
    parser.add_argument('-ds',  required=True, choices=ds_list,
                        help='target dataset', type=str)
    parser.add_argument('-b',  required=False, default=1,
                        help='batch_size used to generate hessian (not needed for make_gradient.py)', type=int)
    parser.add_argument('-t',  required=False, default='train',
                        help='train or test? (not needed for hessian.py)', type=str, choices=['train', 'test'])

    args = parser.parse_args()

    return args

def load_gradients(ds, eps, ub=3, num_images=3200, train_or_test='train'):
    os.chdir(f'{train_or_test}_grad/{ds}_{eps}_eps_{ub}_ub_{num_images}_images')
    files = os.listdir()
    for i, f in enumerate(files):
        if i == 0: gradients = np.load(f)
        else:      gradients = np.vstack((gradients, np.load(f)))

    os.chdir('../..')
    return gradients

