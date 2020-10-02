
base_model_path = 'models/'

eps_to_filename = {
        -1:    None, # Do not load a model that has been trained from scratch
        0:    'nat',
	0.05: 'imagenet_l2_0_05.pt',
	0.25: 'imagenet_l2_0_25.pt',
	1:    'imagenet_l2_1_0.pt',
	3:    'imagenet_l2_3_0.pt',
	4:    'imagenet_linf_4.pt',
	8:    'imagenet_linf_8.pt'
}

# TODO: FILL OUT YOUR DATA PATH BELOW
base_data_path = '/scratch/data/'

