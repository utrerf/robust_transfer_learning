import os
from cox.store import Store
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-o',  required=False, default='log_summary.csv', help='output file', type=str)

args = parser.parse_args()

os.chdir('logs')

df = pd.DataFrame()
folder_list = list(os.listdir())
multiple_store_list = []

for folder in folder_list:
    os.chdir(folder)
    store_id_list = [d for d in os.listdir() if os.path.isdir(d)]
    if len(store_id_list) > 1: 
        multiple_store_list.append(folder)
    try:
        s = Store(os.getcwd(), os.getcwd()+"/"+store_id_list[0])
        new_df = s['logs'].df
    except:
        os.chdir("..")
    else:
        new_df['source_eps'] = re.findall(r'source_eps_([a-z,0-9]+)', folder)[0]
        new_df['target_ds'] = re.findall(r'target_dataset_([a-z,0-9]+)', folder)[0]
        new_df['num_training_images'] = re.findall(r'num_training_images_([a-z,0-9,-]+)', folder)[0]
        new_df['unfrozen_blocks'] = re.findall(r'unfrozen_blocks_([a-z,0-9]+)', folder)[0]
        new_df['seed'] = re.findall(r'seed_([a-z,0-9]+)', folder)[0]

        df = df.append(new_df)
        s.close()

        os.chdir("..")

os.chdir("..")
df.to_csv(args.o)
