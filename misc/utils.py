# -*- coding: utf-8 -*-

import os
import logging
import random
import numpy as np
import json

import torch

from attrdict import AttrDict


def run_scandir(dir_path, patterns):    # dir: str, pattern: list
    sub_dirs, file_paths = list(), list()

    for f in os.scandir(dir_path):
        if f.is_dir():
            sub_dirs.append(f.path)
            pass
        elif f.is_file():
            #if os.path.splitext(f.name)[1].lower() in ext:
            if f.name.split(os.sep)[-1] in patterns:
                file_paths.append(f.path)
                pass
            pass
        pass

    for dir in list(sub_dirs):
        sf, f = run_scandir(dir, patterns)
        sub_dirs.extend(sf)
        file_paths.extend(f)
        pass

    return sub_dirs, file_paths


def run_scandir_re(dir_path, pattern):    # dir: str, pattern: compiled regular-expression
    sub_dirs, file_paths = list(), list()

    for f in os.scandir(dir_path):
        if f.is_dir():
            sub_dirs.append(f.path)
            pass
        elif f.is_file():
            file_name = f.name.split(os.sep)[-1]

            if pattern.match(file_name):
                file_paths.append(f.path)
                pass
            pass
        pass

    for dir in list(sub_dirs):
        sf, f = run_scandir_re(dir, pattern)
        sub_dirs.extend(sf)
        file_paths.extend(f)
        pass

    return sub_dirs, file_paths


def init_logger():
    logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
           ,datefmt='%Y-%m-%d %H:%M:%S'
           ,level=logging.INFO
           )
    pass

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        pass
    pass


def load_config(config_file_path):
    # Read from config file and make args
    with open(config_file_path, 'r') as f:
        config = AttrDict(json.load(f))
        pass
    return config


def split_by_sep(alist, sep):
    t_list = list()
    sub_list = list()

    for elm in alist:
        if elm == sep:
            if sub_list:
                t_list.append(sub_list)
                sub_list = list()
                pass
            pass
        else:
            sub_list.append(elm)
        pass
    if sub_list:
        t_list.append(sub_list)
        pass
    return t_list



def get_gpu(gpu_id=None, print_gpu_info=False):
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()
            device = torch.device(gpu_id)
            pass
        else:
            n_gpus = torch.cuda.device_count()
            if gpu_id < 0 or n_gpus <= gpu_id:
                gpu_id = None
                device = 'cpu'
                pass
            else:
                device = torch.device(f'cuda:{gpu_id}')
                pass
            pass

        gpu_name = torch.cuda.get_device_name(gpu_id)

        if print_gpu_info:
            print(f'gpu available:{gpu_available}, id:{gpu_id},  gpu name:{gpu_name}')
            pass
        pass
    else:
        gpu_id = None
        device = 'cpu'

        if print_gpu_info:
            print(f'gpu not available:')
            pass
        pass

    if print_gpu_info:
        print(f'torch.device({gpu_id}) --> device:"{device}"(type:{type(device)})')
        pass

    return device



def main():
    pass


if __name__ == '__main__':
    main()
    pass
