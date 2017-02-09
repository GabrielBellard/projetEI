import pickle
import os

directory = 'dumps/'
if not os.path.exists(directory):
    os.makedirs(directory)

def pickle_file(filename, obj):
    with open(directory + filename, 'wb') as f:
        pickle.dump(obj, f)


def unpickle_file(filename):
    with open(directory + filename, 'rb') as f:
        return pickle.load(f)