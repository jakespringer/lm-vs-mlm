import os
import sys
import numpy as np
import itertools
import json

# Helper
def product(d):
    def list_or_id(item):
        if not isinstance(item, list):
            return [item]
        else:
            return item

    keys = list(d.keys())
    values = itertools.product(*[list_or_id(d[k]) for k in keys])

    for value in values:
        yield dict(zip(keys, value))

def to_args(params):
    return ' '.join([f'--{k} {v}' for k, v in params.items()])

interpreter = 'python'
script_name = 'train_and_eval.py'
script = f'~/projects/next_token_vs_mlm/src/{script_name}'

args = {
    'num_branches': 2,
    'branch_length': 4,
    'num_train': 5000,
    'num_test': 2000,
    'num_vertices': 50,
    'batch_size': 256,
    'num_layer': 6,
    'num_head': 6,
    'num_embd': 384,
    'dropout': 0.05,
    'bias': 0,
    'block_size': 128,
    'lr': 1e-4,
    'wd': 1e-2,
    'mask_ratio': 0.15,
    'num_epochs': 50,
    'objective': 'mlm',
    'device': 'cuda',
}

extra_args = {
    # 'model_path': '',
}

for i, param in enumerate(product(dict(args, **extra_args))):
    launch_command = f'{interpreter} {script} {to_args(param)}'
    print(launch_command)