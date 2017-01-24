# coding=utf-8
# encoding: utf-8

from __future__ import print_function
import string
import scipy.sparse
from sklearn.feature_extraction import FeatureHasher
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
from constants import *
from joblib import Parallel, delayed
import time
from sklearn import svm

from sklearn.feature_extraction import DictVectorizer


def build_dict_feature_hashing_imdb():
    sentences_train = []

    for ff in tqdm.tqdm(glob.glob(os.path.join(path_train_pos, '*.txt')), desc="train pos"):
        with io.open(ff, 'r', encoding='utf-8') as f:
            sentences_train.append(f.readline().strip())

    for ff in tqdm.tqdm(glob.glob(os.path.join(path_train_neg, '*.txt')), desc="train neg"):
        with io.open(ff, 'r', encoding='utf-8') as f:
            sentences_train.append(f.readline().strip())

    sentences_test = []
    for ff in tqdm.tqdm(glob.glob(os.path.join(path_test_pos, '*.txt')), desc="test pos"):
        with io.open(ff, 'r', encoding='utf-8') as f:
            sentences_test.append(f.readline().strip())

    for ff in tqdm.tqdm(glob.glob(os.path.join(path_test_neg, '*.txt')), desc="test neg"):
        with io.open(ff, 'r', encoding='utf-8') as f:
            sentences_test.append(f.readline().strip())

    X_train = build_dic(sentences_train, double_features)
    X_test = build_dic(sentences_test, double_features)

    return X_train, X_test


def build_dict_feature_hashing_mrd():
    sentences_pos = []

    ff = os.path.join(dataset_path_mrd, 'rt-polarity_utf8.pos')

    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences pos"):
            time.sleep(0.001)
            sentences_pos.append(line)

    sentences_neg = []
    ff = os.path.join(dataset_path_mrd, 'rt-polarity_utf8.neg')
    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences neg"):
            time.sleep(0.001)
            sentences_neg.append(line)

    sentences = sentences_pos + sentences_neg

    X = build_dic(sentences, double_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, [1] * len(sentences_pos) + [0] * len(sentences_neg), test_size=0.4,
        random_state=58)

    return X_train, X_test, y_train, y_test


def build_dic(sentences, double_features):
    hasher = HashingVectorizer(n_features=2 ** 18,
                               stop_words='english',
                               norm=None)
    vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

    if sentiwordnet:
        polarity_arr2 = []

        polarity_arr = Parallel(n_jobs=num_cores)(delayed(get_polarity)(sentence, double_features) for
                                                  sentence in tqdm.tqdm(sentences, desc="compute polarity"))

        [polarity_arr2.append(dict_) for list in polarity_arr for dict_ in list]

    sentences_stem = []
    sentences_stem2 = []
    # sentences_stem = stemmering_sentences(sentences)

    sentences_stem = Parallel(n_jobs=num_cores)(delayed(stemmering_sentences)(sentence) for
                                                sentence in tqdm.tqdm(sentences, desc="stem"))

    if sentiwordnet:
        feature_hasher = FeatureHasher()
        X_polarity = feature_hasher.fit_transform(polarity_arr2)

    # [sentences_stem2.append(liste) for liste in sentences_stem]


    sentences_stem2 = [' '.join(term) for term in sentences_stem]

    X_sentences = vectorizer.fit_transform(sentences_stem2)

    if sentiwordnet:
        X = scipy.sparse.hstack([X_sentences, X_polarity])
        return X

    else:
        return X_sentences


def get_polarity(sentence, double_features):
    # sentences_stem_ = do_stemming(sentence)

    polarity_arr = compute_polarity(sentence, double_features)

    # polarity_arr.append(polarity_arr_)

    return polarity_arr


def compute_polarity(sentence, double_features):

    polarity_arr = []
    polarity_dic = {}
    if double_features:
        global_pos = 0
        global_neg = 0
    else:
        global_pol = 0
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
                if double_features:
                    global_pos = 0.0
                    global_neg = 0.0
                else:
                    global_pol = 0
            else:
                word_pos, word_neg = svn.get_score_word(word, pos_tag)
                if double_features:
                    global_pos += word_pos
                    global_neg += word_neg
                else:
                    global_pol = word_pos - word_neg
    if double_features:
        polarity_dic["pos"] = global_pos
        polarity_dic["neg"] = global_neg
    else:
        polarity_dic["pol"] = global_pol
    polarity_arr.append(polarity_dic)

    return polarity_arr


def stemmering_sentences(sentence):
    sentences_stem = []
    # Remove punctuation, stopword and then stemmering
    punctuation = set(string.punctuation)
    stemmer = nltk.PorterStemmer()

    tmp = sentence

    doc = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(tmp) if
           (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english'))]
    #sentences_stem.append(doc)

    return doc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model you want to use')
    parser.add_argument('-d', '--dataset', help='which dataset imdb or mrd', required=True, nargs=1, type=str)
    parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet', action='store_true')
    parser.add_argument('-np', '--number_polarity', help='if 2 features pos and neg or global', required=False,
                        type=int, nargs=1)
    parser.add_argument('-j', '--joblib', help='number of jobs for parallelism', required=False, type=int,
                        default=num_cores)
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    n_polarity = 0
    if args.sentiwordnet is True and args.number_polarity is None:
        parser.error("-s, --sentiwordnet requires --np 1|2 ")

    if args.sentiwordnet is False:
        n_polarity = 0
        sentiwordnet = False

    if args.sentiwordnet and 2 in args.number_polarity:
        double_features = True
        n_polarity = 2
        sentiwordnet = True

    elif args.sentiwordnet and 1 in args.number_polarity:
        double_features = False
        n_polarity = 1
        sentiwordnet = True

    if args.joblib != num_cores:
        num_cores = args.j

    if "imdb" in str(args.dataset):
        X_train, X_test = build_dict_feature_hashing_imdb()
        n_train = X_train.shape[0] / 2
        y_train = [1] * n_train + [0] * n_train

        n_test = X_test.shape[0] / 2

        y_test = [1] * n_test + [0] * n_test

        print("pickle file : "+'test_imdb_' + str(n_polarity) + '.pkl')
        pickle_file('test_imdb_' + str(n_polarity) + '.pkl', (X_test, y_test))

        print("Fitting SVM")
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)

        print('Saving model : '+'train_imdb_' + str(n_polarity) + '.pkl')
        pickle_file('train_imdb_' + str(n_polarity) + '.pkl', clf)

    elif "mrd" in str(args.dataset):
        X_train, X_test, y_train, y_test = build_dict_feature_hashing_mrd()

        print("pickle file : " + 'test_mrd_' + str(n_polarity) + '.pkl')
        pickle_file('test_mrd_' + str(n_polarity) + '.pkl', (X_test, y_test))

        print("Fitting SVM")
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)

        print('Saving model : ' + 'train_imdb_' + str(n_polarity) + '.pkl')
        pickle_file('train_mrd_' + str(n_polarity) + '.pkl', clf)

    else:
        parser.error("-d, --dataset requires imdb or mrd")
