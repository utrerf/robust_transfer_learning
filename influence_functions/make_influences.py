import os
import numpy as np
import sys
sys.path.insert(1, '..')
from tools.helpers import get_runtime_inputs_for_influence_functions, load_gradients


args = get_runtime_inputs_for_influence_functions()

eps        = args.e
num_images = args.n
ub         = args.ub
seed       = args.s
ds         = args.ds
b          = args.b
data_type  = args.t
isTrain = (data_type == 'train')

train_gradients = load_gradients(ds, eps, ub, num_images, train_or_test='train')
test_gradients = train_gradients
if isTrain == False:
    test_gradients  = load_gradients(ds, eps, ub, num_images, train_or_test='test')

H_inv = np.load(f'h_inverses/H_inv_{ds}_target_{ub}_ub_{eps}_eps_{num_images}_images_{b}_batch_size.npy')

influences = test_gradients @ H_inv @ train_gradients.T

np.save(f'influences/{ds}_target_{ub}_ub_{eps}_eps_{num_images}_images_{b}_batch_size_{data_type}', influences)
