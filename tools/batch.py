import os
import time
import subprocess
import shlex
import os
from delete_big_files import deleteBigFilesFor1000experiment

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

train_dir = '/home/eecs/erichson/arnn_transfer/new_rob/robust_transfer_learning'

sample_size_to_number_of_seeds_epochs_and_log_freq = {
	100  : (10, 100, 20),
	200  : (10, 100, 20),
	400  : (10, 100, 20),
	800  : (10, 100, 20),
	1600 : (10, 100, 20),
	3200 : (10, 150, 10),
	6400 : (10, 150, 10),
	12800: (5,  150, 10),
	25600: (5,  150, 10),
	-1   : (1,  150, 10),
}

target_ds_list = ['cifar100', 'cifar10', 'svhn', 'kmnist', 'fmnist', 'mnist']
eps_levels = [0, 3, 4, 8]
num_unfrozen_blocks_list = [0, 1, 3, 6]	


polling_delay_seconds = 1
concurrent_commands = 3
commands_to_run = []

def poll_process(process):
	time.sleep(polling_delay_seconds)
	return process.poll()

for t in target_ds_list:
	for ub in num_unfrozen_blocks_list:
		for n, tup in sample_size_to_number_of_seeds_epochs_and_log_freq.items():
			num_seeds, ne, li = tup
			seed_list = [20000000 + 100000*(i) for i in range(num_seeds)]
			for s in seed_list:
				for e in eps_levels:
					command = f'python train.py -e {e} -t {t} -ub {ub} -n {n} -s {s} -ne {ne} -li {li}'
					commands_to_run.append(command)

for start_idx in range(0, len(commands_to_run), concurrent_commands):
	os.chdir(train_dir)
	processes = []
	for i in range(start_idx, min(len(commands_to_run), start_idx + concurrent_commands)):
		processes.append(subprocess.Popen(shlex.split(commands_to_run[i])))
		print(f'Starting command: {commands_to_run[i]}')
	
	for process in processes:
		while poll_process(process) is None:
			pass
	
	deleteBigFilesFor1000experiment()


