import string

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import *
import glob
import os
import io
import nltk
import SentiWordNet as svn
from nltk.tokenize import RegexpTokenizer

dataset_path_imdb = '/home/gabz/Documents/projetEI/aclImdb/'
dataset_path_mrd = '/home/gabz/Documents/projetEI/rt-polaritydata/'


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

	hasher = HashingVectorizer(n_features=2 ** 18,
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
	ff = "rt-polarity_utf8.pos"
	with io.open(ff, 'r', encoding='UTF-8') as f:
		for line in f:
			sentences_pos.append(line)

	sentences_neg = []
	ff = "rt-polarity_utf8.neg"
	with io.open(ff, 'r', encoding='UTF-8') as f:
		for line in f:
			sentences_neg.append(line)
	os.chdir(currdir)

	# for item in nltk.bigrams(tweetString.split()):
	# 	bigramFeatureVector.append(' '.join(item))


	hasher = HashingVectorizer(n_features=2 ** 18,
							   stop_words='english', non_negative=True,
							   norm=None, binary=False, ngram_range=(1,3))

	vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

	sentences = sentences_pos + sentences_neg

	for sentence in sentences:
		print("pour la phrase " + sentence)
		global_pos = 0
		global_neg = 0
		sentence = nltk.word_tokenize(sentence)
		for word in sentence :
			if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words('english')):
				word_pos, word_neg = svn.get_score_word(word)
				global_pos += word_pos
				global_neg += word_neg
		print("pos " + str(global_pos))
		print("neg " + str(global_neg))

	sentences = stemmering_sentences_mrd(sentences)

	print(sentences)

	X_sentences = vectorizer.fit_transform([' '.join(term) for term in sentences])





	# X_train, X_test, y_train, y_test = train_test_split(
	# 	sentences, [1] * len(sentences_pos) + [0] * len(sentences_neg), test_size=0.4,
	# 	random_state=58)

	# X_train = vectorizer.fit_transform([' '.join(term) for term in X_train])
	# X_test = vectorizer.fit_transform([' '.join(term) for term in X_test])

	# return X_train, X_test, y_train, y_test


def stemmering_sentences(sentences_train, sentences_test):
	# Remove punctuation, stopword and then stemmering
	punctuation = set(string.punctuation)
	stemmer = nltk.PorterStemmer()
	for i in range(len(sentences_train)):
		tmp = sentences_train[i]
		# tmp = unicode(tmp, errors='ignore')
		doc = [stemmer.stem(word) for word in nltk.word_tokenize(tmp) if
			   (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english'))]
		sentences_train[i] = doc

	for i in range(len(sentences_test)):
		tmp = sentences_test[i]
		tmp = unicode(tmp, errors='ignore')
		doc = [stemmer.stem(word) for word in nltk.word_tokenize(tmp) if
			   (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english'))]
		sentences_test[i] = doc


def stemmering_sentences_mrd(sentences):
	sentences_pos_stem = []
	# Remove punctuation, stopword and then stemmering
	punctuation = set(string.punctuation)
	stemmer = nltk.PorterStemmer()
	for i in range(len(sentences)):
		tmp = sentences[i]

		doc = [stemmer.stem(word) for word in tmp if
			   (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english'))]
		sentences_pos_stem.append(doc)

	return sentences_pos_stem


def main():

	imdb = False

	if imdb:
		X_train, X_test = build_dict_feature_hashing_imdb(os.path.join(dataset_path_imdb, 'train'),
														  os.path.join(dataset_path_imdb, 'test'))
		n_train = X_train.shape[0] / 2
		y_train = [1] * n_train + [0] * n_train

		n_test = X_test.shape[0] / 2

		y_test = [1] * n_test + [0] * n_test

	else:
		X_train, X_test, y_train, y_test = build_dict_feature_hashing_mrd(dataset_path_mrd)


	# if imdb:
	# 	pickle_file('train_imdb.pkl', (X_train, y_train))
	# 	pickle_file('test_imdb.pkl', (X_test, y_test))
	# else:
	# 	pickle_file('train_mrd_stem.pkl', (X_train, y_train))
	# 	pickle_file('test_mrd_stem.pkl', (X_test, y_test))


if __name__ == '__main__':
	main()
