import pickle

def pickle_file(filename, obj):
    with open('dumps/' + filename, 'wb') as f:
        pickle.dump(obj, f)


def unpickle_file(filename):
    with open('dumps/' + filename, 'rb') as f:
        return pickle.load(f)