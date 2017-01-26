import os
import multiprocessing
import argparse

dataset_path_imdb = 'aclImdb/'
path_train = os.path.join(dataset_path_imdb, 'train')
path_train_pos = os.path.join(path_train, 'pos')
path_train_neg = os.path.join(path_train, 'neg')

path_test = os.path.join(dataset_path_imdb, 'test')
path_test_pos = os.path.join(path_test, 'pos')
path_test_neg = os.path.join(path_test, 'neg')

dataset_path_mrd = 'rt-polaritydata/'


num_cores = multiprocessing.cpu_count()