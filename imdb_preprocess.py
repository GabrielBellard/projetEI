from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from utils import *
import glob
import os

dataset_path = '/home/gabz/Documents/projetEI/aclImdb/'

def build_dict_feature_hashing(path_train, path_test):
	sentences_train = []
	currdir = os.getcwd()
	os.chdir('%s/pos/' % path_train)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			sentences_train.append(f.readline().strip())
	os.chdir('%s/neg/' % path_train)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			sentences_train.append(f.readline().strip())
	os.chdir(currdir)

	sentences_test = []
	currdir = os.getcwd()
	os.chdir('%s/pos/' % path_test)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			sentences_test.append(f.readline().strip())
	os.chdir('%s/neg/' % path_test)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			sentences_test.append(f.readline().strip())
	os.chdir(currdir)


	hasher = HashingVectorizer(n_features=2**18,
							   stop_words='english', non_negative=True,
							   norm=None, binary=False)
	vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

	X_train = vectorizer.fit_transform(sentences_train)
	X_test = vectorizer.fit_transform(sentences_test)

	return X_train, X_test



def main():

	path = dataset_path
	X_train, X_test = build_dict_feature_hashing(os.path.join(path, 'train'), os.path.join(path, 'test'))
	print X_train.shape
	print X_test.shape

	n_train = X_train.shape[0]/2
	train_y = [1] * n_train  + [0] * n_train

	n_test = X_test.shape[0] / 2

	test_y = [1] * n_test + [0] * n_test

	pickle_file('train.pkl', (X_train, train_y))
	pickle_file('test.pkl', (X_test, test_y))



if __name__ == '__main__':
	main()
