import os
import time
import subprocess
import shlex
import os
from delete_big_files import deleteBigFilesFor1000experiment

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

train_dir = '/home/ubuntu/robust_transfer_learning/' 

sample_size_to_number_of_seeds_epochs_and_log_freq = {
    400  : (3, 150, 5),
    1600 : (2, 150, 5),
    6400 : (1, 150, 5),
    25600: (1, 150, 5),
    -1   : (1, 150, 5),
}

target_ds_list = ['food101']
eps_levels = [0, 0.05, 0.25, 1]
num_unfrozen_blocks_list = [3]	


polling_delay_seconds = 1
concurrent_commands = 4
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
                    command = f'python train.py -e {e} -t {t} -ub {ub} -n {n} -s {s} -ne {ne} -li {li} -d True'
                    commands_to_run.append(command)

for start_idx in range(0, len(commands_to_run), concurrent_commands):
    os.chdir(train_dir)
    processes = []
    rng = range(start_idx, min(len(commands_to_run), start_idx + concurrent_commands))
    print(rng)
    for i in rng:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i%4)
        processes.append(subprocess.Popen(shlex.split(commands_to_run[i])))
        print(f'Starting command: {commands_to_run[i]}')
        
    for process in processes:
        while poll_process(process) is None:
            pass
	
    deleteBigFilesFor1000experiment()


