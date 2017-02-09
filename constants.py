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

dataset_path_spd = 'rt-polaritydata/'


num_cores = multiprocessing.cpu_count()
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 5
n_iterations = 1
window_size = 5


vocab_dim_IMDB = 50
maxlen_IMDB = 400
n_exposures_IMDB = 30
batch_size_IMDB = 200

vocab_dim_SPD = 32
maxlen_SPD = 40
n_exposures_SPD = 2
batch_size_SPD = 100


n_polarity = 0
sentiwordnet = False
double_features = False
hashing_trick = False
negation = False
feature_selection = False
