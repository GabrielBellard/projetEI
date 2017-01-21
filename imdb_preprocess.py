import string

import scipy.sparse
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
import tqdm
from sklearn.feature_extraction import DictVectorizer
import metacritic

dataset_path_imdb = 'aclImdb/'
dataset_path_mrd = 'rt-polaritydata/'


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

    sentences = sentences_pos + sentences_neg

    X = build_dic(sentences)

    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, [1] * len(sentences_pos) + [0] * len(sentences_neg), test_size=0.4,
        random_state=58)

    # X_train = vectorizer.fit_transform([' '.join(term) for term in X_train])
    # X_test = vectorizer.fit_transform([' '.join(term) for term in X_test])

    return X_train, X_test, y_train, y_test


def build_dic(sentences):
    hasher = HashingVectorizer(n_features=2 ** 18,
                               stop_words='english', non_negative=True,
                               norm=None, binary=False)
    vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])
    polarity_arr = []
    for sentence in tqdm.tqdm(sentences):
        polarity_dic = {}
        global_pos = 0
        global_neg = 0
        sentence = nltk.word_tokenize(sentence)
        for word in sentence:
            if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words('english')):

                pos_tag = str(nltk.tag.pos_tag([word])[0][1]).lower()

                if pos_tag.startswith("n"):
                    pos_tag = 'n'
                elif pos_tag.startswith("v"):
                    pos_tag = 'v'
                elif pos_tag.startswith("j"):
                    pos_tag = 'a'
                elif pos_tag.startswith("r"):
                    pos_tag = 'r'
                else:
                    pos_tag = None

                if pos_tag is None:
                    global_pos = 0.0
                    global_neg = 0.0
                else:
                    word_pos, word_neg = svn.get_score_word(word, pos_tag)
                    global_pos += word_pos
                    global_neg += word_neg

        polarity_dic["pos"] = global_pos
        polarity_dic["neg"] = global_neg
        polarity_arr.append(polarity_dic)

    # polarity_arr = []
    # for sentence in tqdm.tqdm(sentences):
    #     polarity_dic = {}
    #     global_pol = 0
    #     sentence = nltk.word_tokenize(sentence)
    #     for word in sentence:
    #         if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words('english')):
    #
    #             pos_tag = str(nltk.tag.pos_tag([word])[0][1]).lower()
    #
    #             if pos_tag.startswith("n"):
    #                 pos_tag = 'n'
    #             elif  pos_tag.startswith("v"):
    #                 pos_tag = 'v'
    #             elif pos_tag.startswith("j"):
    #                 pos_tag = 'a'
    #             elif pos_tag.startswith("r"):
    #                 pos_tag = 'r'
    #             else:
    #                 pos_tag = None
    #
    #             if pos_tag is None:
    #                  global_pol = 0
    #             else:
    #                 word_pos, word_neg = svn.get_score_word(word, pos_tag)
    #                 global_pol = word_pos - word_neg
    #
    #     polarity_dic["pol"] = global_pol
    #     polarity_arr.append(polarity_dic)
    v = DictVectorizer()
    X_polarity = v.fit_transform(polarity_arr)
    sentences = stemmering_sentences_mrd(sentences)
    X_sentences = vectorizer.fit_transform([' '.join(term) for term in sentences])
    X = scipy.sparse.hstack([X_sentences, X_polarity])
    return X


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
        doc = [stemmer.stem(word) for word in nltk.word_tokenize(tmp) if
               (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english'))]
        sentences_test[i] = doc


def stemmering_sentences_mrd(sentences):
    sentences_pos_stem = []
    # Remove punctuation, stopword and then stemmering
    punctuation = set(string.punctuation)
    stemmer = nltk.PorterStemmer()
    for i in tqdm.tqdm(range(len(sentences))):
        tmp = sentences[i]

        doc = [stemmer.stem(word) for word in nltk.word_tokenize(tmp) if
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

    if imdb:
        pickle_file('train_imdb.pkl', (X_train, y_train))
        pickle_file('test_imdb.pkl', (X_test, y_test))
    else:
        pickle_file('train_mrd_sentiword_postag.pkl', (X_train, y_train))
        pickle_file('test_mrd_sentiword _postag.pkl', (X_test, y_test))


if __name__ == '__main__':
    main()
