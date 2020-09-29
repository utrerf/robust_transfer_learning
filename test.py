import argparse
import tools.custom_datasets as custom_datasets
from tools.custom_datasets import ImageNetTransfer, name_to_dataset
import tools.constants as constants
from cox.utils import Parameters
import cox.store
import os
from robustness.model_utils import make_and_restore_model
import robustness.train as train

def get_input_args():
  eps_list            = list(constants.eps_to_filename.keys())
  target_dataset_list = list(custom_datasets.name_to_dataset.keys())
  norm_list = ['inf', '2']

  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('-p',   required=False, default=0,         help='resume path for the model', type=str)
  parser.add_argument('-out', required=False, default=f'{os.getcwd()}/test_results',         help='resume path for the model', type=str)
  parser.add_argument('-ds',  required=False, default='cifar10', help='name of the target dataset', type=str, choices=target_dataset_list)
  parser.add_argument('-e',   required=False, default=0,         help='test epsilon',  type=float)
  parser.add_argument('-pgd', required=False, default=128,       help='number of pgd steps for testing',  type=int)
  parser.add_argument('-lp',  required=False, default='2',       help='norm of the lp constraint',  type=str, choices=norm_list)

  input_args = parser.parse_args()

  return input_args

input_args = get_input_args()

dataset = ImageNetTransfer(name=input_args.ds,
                     data_path=f'{constants.base_data_path}{input_args.ds}',
                     num_transf_classes=name_to_dataset[input_args.ds]['num_classes'],
                     downscale=False, **name_to_dataset[input_args.ds])

_, test_loader = dataset.make_loaders(8, 128)

if not os.path.exists(input_args.out): os.mkdir(input_args.out)
store = cox.store.Store(f'{input_args.out}/dataset_{input_args.ds}_eps_{input_args.e}_PGD_{input_args.pgd}_norm_{input_args.lp}')

model, _ = make_and_restore_model(arch='resnet50', dataset=dataset, parallel=False, resume_path=input_args.p, pytorch_pretrained=False)

test_args = {
   'adv_eval': input_args.e > 0,
   'use_best': True,
   'random_restarts': False,
   'out_dir': "train_out",
   'constraint': input_args.lp, # L-inf PGD
   'eps': input_args.e, # Epsilon constraint (L-inf norm)
   'attack_lr': 2.5*(input_args.e/input_args.pgd), # Learning rate for PGD
   'attack_steps': input_args.pgd, # Number of PGD steps
}

test_args = Parameters(test_args)

train.eval_model(test_args, model, test_loader, store)
