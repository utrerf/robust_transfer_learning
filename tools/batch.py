import os
import time
import subprocess
import shlex
import os
from delete_big_files import deleteBigFilesFor1000experiment

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

train_dir = '/home/eecs/erichson/arnn_transfer/robust_transfer_learning'


#eps_levels = [0, 3, 4]
#num_unfrozen_blocks_list = [0, 1, 3]
sample_size = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, -1]
# sample_size_to_seeds = 
# {
# 	100   : [10000000 + 100000* for i in range(25)],
# 	200   : [10000000 + 100000*i for i in range(25)],
# 	400   : [10000000 + 100000*i for i in range(25)],
# 	800   : [10000000 + 100000*i for i in range(25)],
# 	1600  : [10000000 + 100000*i for i in range(25)],
# 	3200  : [10000000 + 100000*i for i in range(25)],
# 	6400  : [10000000 + 100000*i for i in range(25)],
# }

num_seeds = 5
seed_list = [10000000 + 100000*(i) for i in range(num_seeds)] 

target_ds_list = ['svhn']	
eps_levels = [0, 3, 4]
num_unfrozen_blocks_list = [1]	


polling_delay_seconds = 1
concurrent_commands = 2
commands_to_run = []

def poll_process(process):
	time.sleep(polling_delay_seconds)
	return process.poll()

for t in target_ds_list:
	for e in eps_levels:
		for ub in num_unfrozen_blocks_list:
			for n in sample_size:
				# seed_list = sample_size_to_seeds[n]
				for s in seed_list:
					command = f'python train.py -e {e} -t {t} -ub {ub} -n {n} -s {s} '
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

