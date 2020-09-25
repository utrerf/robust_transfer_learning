import tools.helpers as helpers
from tools.custom_datasets import name_to_dataset, make_dataset
from robustness import train
import cox.store


var_dict = helpers.get_runtime_inputs()

helpers.set_seeds(var_dict)

# do we transfer?
is_Transfer = False
if var_dict['source_eps'] > -1: is_Transfer = True

# do we grab an imagenet pretrained model?
pretrained = False
if var_dict['source_eps'] == 0: pretrained = True

# get dataset class
dataset = make_dataset(var_dict)

model = helpers.load_model(var_dict, is_Transfer, pretrained, dataset)

model = helpers.change_linear_layer_out_features(model, var_dict, dataset)

if is_Transfer: model = helpers.re_init_and_freeze_blocks(model, var_dict)

subset = var_dict['num_training_images']
if var_dict['num_training_images'] == -1: subset = None
train_loader, test_loader = dataset.make_loaders(workers=var_dict['num_workers'], batch_size=var_dict['batch_size'],
                                                 subset=subset, subset_seed=var_dict['seed'])

out_store  = helpers.make_out_store(var_dict)
train_args = helpers.make_train_args(var_dict)

helpers.print_details(model, var_dict, train_args)

train.train_model(train_args, model, (train_loader, test_loader), store=out_store)
pass
