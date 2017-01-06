from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from utils import *
import glob
import os
import io

dataset_path_imdb = '/home/gabz/Documents/projetEI/aclImdb/'
dataset_path_mrd= '/home/gabz/Documents/projetEI/rt-polaritydata/'

def build_dict_feature_hashing_imdb(path_train, path_test):
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

def build_dict_feature_hashing_mrd(path):
	sentences_pos = []
	currdir = os.getcwd()
	os.chdir('%s' % path)
	ff = "rt-polarity.pos"
	with io.open(ff, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			sentences_pos.append(line)

	sentences_neg = []
	ff = "rt-polarity.neg"
	with io.open(ff, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			sentences_neg.append(line)
	os.chdir(currdir)

	hasher = HashingVectorizer(n_features=2**18,
							   stop_words='english', non_negative=True,
							   norm=None, binary=False)
	vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

	sentences = sentences_pos + sentences_neg
	X = vectorizer.fit_transform(sentences)

	X_train =X[:int(X.shape[0]*0.75)]
	X_test = X[int(X.shape[0]*0.75):]

	return X_train, X_test


def main():

	imdb = False
	if imdb :
		X_train, X_test = build_dict_feature_hashing_imdb(os.path.join(dataset_path_imdb, 'train'),
														  os.path.join(dataset_path_imdb, 'test'))
	else:
		X_train, X_test = build_dict_feature_hashing_mrd(dataset_path_mrd)

	print X_train.shape
	print X_test.shape

	n_train = X_train.shape[0]/2
	train_y = [1] * n_train  + [0] * n_train

	n_test = X_test.shape[0] / 2

	test_y = [1] * n_test + [0] * n_test

	if imdb:
		pickle_file('train_imdb.pkl', (X_train, train_y))
		pickle_file('test_imdb.pkl', (X_test, test_y))
	else:
		pickle_file('train_mrd.pkl', (X_train, train_y))
		pickle_file('test_mrd.pkl', (X_test, test_y))



if __name__ == '__main__':
	main()
