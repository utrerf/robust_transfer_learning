import tools.helpers as helpers
from robustness import train
import cox.store

var_dict = helpers.get_runtime_inputs()

helpers.set_seeds(var_dict)

model = helpers.load_model(var_dict)
model = helpers.change_linear_layer_out_features(model, var_dict)
model = helpers.re_init_and_freeze_blocks(model, var_dict)

train_set, test_set       = helpers.make_train_and_test_set(var_dict)
train_loader, test_loader = helpers.make_data_loaders(train_set, test_set, var_dict)

out_store  = helpers.make_out_store(var_dict)
train_args = helpers.make_train_args(var_dict)

helpers.print_details(model, var_dict, train_args)

train.train_model(train_args, model, (train_loader, test_loader), store=out_store)
pass
