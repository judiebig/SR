import pickle
import argparse
import collections
import tqdm
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from data import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset: yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')

opt = parser.parse_args()

n_node = 37484


def main():
    # preliminaries
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename="result/log.txt", filemode='w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_print(json.dumps(opt.__dict__, indent=4))

    ''' ---------process data------------- '''


if __name__ == "__main__":
    main()
