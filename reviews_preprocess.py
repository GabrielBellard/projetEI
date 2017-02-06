# coding=utf-8
# encoding: utf-8

from __future__ import print_function
import string

import scipy.sparse
from bs4 import BeautifulSoup
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import *
import glob
import io
import nltk
import SentiWordNet as svn
import tqdm
from constants import *
from joblib import Parallel, delayed
import time
from sklearn import svm
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def build_dict_feature_vectorizer_imdb(double_features):
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

    X_train, vectorizer_fitted = build_dic(sentences_train, double_features)
    X_test, _ = build_dic(sentences_test, double_features, vectorizer_fitted)

    pickle_file("X_train.pkl", X_train)
    pickle_file("X_test.pkl", X_test)

    X_train = unpickle_file("X_train.pkl")
    X_test = unpickle_file("X_test.pkl")

    print(X_train.shape)
    print(X_test.shape)
    n = X_train.shape[0] / 2
    y_train = [1] * n + [0] * n
    y_test = [1] * n + [0] * n

    pca = TruncatedSVD(n_components=2, algorithm="arpack").fit(X_train)
    data2D = pca.transform(X_train)
    plt.scatter(data2D[:, 0], data2D[:, 1], c=y_train)
    plt.show()

    if hashing_trick:
        fselect = SelectKBest(chi2, k=200000)
    else:
        if negation:
            fselect = SelectKBest(chi2, k=200000)
        else:
            fselect = SelectKBest(chi2, k=200000)

    X_train = fselect.fit_transform(X_train, y_train)

    X_test = fselect.transform(X_test)

    return X_train, X_test, y_train, y_test


def build_dict_feature_vectorizer_mrd(double_features):
    sentences_pos = []

    ff = os.path.join(dataset_path_mrd, 'rt-polarity_utf8.pos')

    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences pos"):
            # time.sleep(0.001)
            sentences_pos.append(line)

    sentences_neg = []
    ff = os.path.join(dataset_path_mrd, 'rt-polarity_utf8.neg')
    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences neg"):
            # time.sleep(0.001)
            sentences_neg.append(line)

    sentences = sentences_pos + sentences_neg

    y = [1] * (len(sentences_pos)) + [0] * (len(sentences_neg))

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=58)

    X_train, vectorizer = build_dic(sentences_train, double_features)
    X_test, _ = build_dic(sentences_test, double_features, vectorizer)

    pickle_file("mrd_train.pkl", X_train)
    pickle_file("mrd_test.pkl", X_test)

    X_train = unpickle_file("mrd_train.pkl")
    X_test = unpickle_file("mrd_test.pkl")

    pca = TruncatedSVD (n_components=2).fit(X_train)
    data2D = pca.transform(X_train)
    plt.scatter(data2D[:, 0], data2D[:, 1], c = y_train)
    plt.show()

    if hashing_trick:
        fselect = SelectKBest(chi2, k=9500)
    else:
        if negation:
            fselect = SelectKBest(chi2, k=9500)
        else:
            fselect = SelectKBest(chi2, k=8500)



    X_train = fselect.fit_transform(X_train, y_train)

    X_test = fselect.transform(X_test)

    # f = np.asarray(fselect.get_feature_names())[chi2.get_support()]

    return X_train, X_test, y_train, y_test


def build_dic(sentences, double_features, vectorizer_fitted=None):
    do_negation = negation
    if hashing_trick and vectorizer_fitted is None:
        hasher = HashingVectorizer(norm=None, non_negative=True)

        vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

    elif not hashing_trick and vectorizer_fitted is None:
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 6),
                                     sublinear_tf=True)

    elif vectorizer_fitted is not None:
        vectorizer = vectorizer_fitted

    if sentiwordnet:
        polarity_arr2 = []

        polarity_arr = Parallel(n_jobs=num_cores)(delayed(get_polarity)(sentence, double_features) for
                                                  sentence in tqdm.tqdm(sentences, desc="compute polarity"))

        [polarity_arr2.append(dict_) for list in polarity_arr for dict_ in list]

        feature_hasher = FeatureHasher(non_negative=True)
        X_polarity = feature_hasher.fit_transform(polarity_arr2)

    sentences_stem = Parallel(n_jobs=num_cores)(delayed(preprocessing_sentences)(sentence, do_negation) for
                                                sentence in tqdm.tqdm(sentences, desc="preprocessing"))

    sentences_stem2 = [' '.join(term) for term in sentences_stem]

    start = time.time()
    if vectorizer_fitted is not None:
        X_sentences = vectorizer.transform(sentences_stem2)
    else :
        X_sentences = vectorizer.fit_transform(sentences_stem2)
    end = time.time()
    elapsed = end - start
    print("temps de vectorisation : " + str(elapsed))

    if sentiwordnet:
        X = scipy.sparse.hstack([X_sentences, X_polarity])
        return X, vectorizer

    else:
        return X_sentences, vectorizer


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

        if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words('english')) and word != 'br':

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


def preprocessing_sentences(sentence, do_negation):
    # Remove punctuation, stopword and then stemmering
    punctuation = set(string.punctuation)
    stemmer = nltk.PorterStemmer()

    sentence = BeautifulSoup(sentence, "lxml").get_text()
    # sentence = nltk.re.sub("[^a-zA-Z]", " ", sentence)

    tmp = sentence
    doc = []
    negation = False
    word_nega = ""

    for word in nltk.word_tokenize(tmp):
        if do_negation:
            if word in ["not", "n't", "no"]:
                negation = True
                word_nega = word + "_"

            if word in punctuation or word == "but":
                negation = False

        if (word not in punctuation) and (word not in nltk.corpus.stopwords.words('english')):
            word_stem = stemmer.stem(word.lower())

            if negation:
                negated = word_nega.upper() + word_stem
                doc.append(negated)
            else:
                doc.append(word_stem)

    return doc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model you want to use')
    parser.add_argument('-d', '--dataset', help='which dataset imdb or mrd', required=True, nargs=1, type=str)
    parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet and precise number of features', required=False,
                        type=int)
    parser.add_argument('-m', '--model', help='which model svm or lstm', required=True, nargs=1, type=str)
    parser.add_argument('-j', '--joblib', help='number of jobs for parallelism', required=False, type=int,
                        default=num_cores)
    parser.add_argument('-hash', '--hash', help='whether use hashing trick', required=False, action='store_true')
    parser.add_argument('-n', '--negation', help='whether do negation(bad result)', required=False, action='store_true')
    args = parser.parse_args()

    n_polarity = 0
    sentiwordnet = False
    double_features = False
    hashing_trick = False
    negation = False

    if args.sentiwordnet == 2:
        double_features = True
        n_polarity = 2
        sentiwordnet = True

    elif args.sentiwordnet == 1:
        double_features = False
        n_polarity = 1
        sentiwordnet = True

    if args.joblib != num_cores:
        num_cores = args.joblib

    if "lstm" in args.model:
        model = "lstm"
    else:
        model = "svm"

    if args.hash:
        hashing_trick = True

    if args.negation:
        negation = True

    if "imdb" in str(args.dataset):
        X_train, X_test, y_train, y_test = build_dict_feature_vectorizer_imdb(double_features)

        if hashing_trick:
            print("pickle file : " + model + '_test_imdb_' + str(n_polarity) + '_hash.pkl')
            pickle_file(model + '_test_imdb_' + str(n_polarity) + '_hash.pkl', (X_test, y_test))
        else:
            if negation:
                print("pickle file : " + model + '_test_imdb_' + str(n_polarity) + '_n.pkl')
                pickle_file(model + '_test_imdb_' + str(n_polarity) + '_n.pkl', (X_test, y_test))
            else:
                print("pickle file : " + model + '_test_imdb_' + str(n_polarity) + '.pkl')
                pickle_file(model + '_test_imdb_' + str(n_polarity) + '.pkl', (X_test, y_test))

        print("Fitting SVM")
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)

        if hashing_trick:
            print('Saving model : ' + model + '_train_imdb_' + str(n_polarity) + '_hash.pkl')
            pickle_file(model + '_train_imdb_' + str(n_polarity) + '_hash.pkl', clf)
        else:
            if negation:
                print('Saving model : ' + model + '_train_imdb_' + str(n_polarity) + '_n.pkl')
                pickle_file(model + '_train_imdb_' + str(n_polarity) + '_n.pkl', clf)
            else:
                print('Saving model : ' + model + '_train_imdb_' + str(n_polarity) + '.pkl')
                pickle_file(model + '_train_imdb_' + str(n_polarity) + '.pkl', clf)

    elif "mrd" in str(args.dataset):
        X_train, X_test, y_train, y_test = build_dict_feature_vectorizer_mrd(double_features)

        if hashing_trick:
            print("pickle file : " + model + '_test_mrd_' + str(n_polarity) + '_hash.pkl')
            pickle_file(model + '_test_mrd_' + str(n_polarity) + '_hash.pkl', (X_test, y_test))
        else:
            if negation:
                print("pickle file : " + model + '_test_mrd_' + str(n_polarity) + '_n.pkl')
                pickle_file(model + '_test_mrd_' + str(n_polarity) + '_n.pkl', (X_test, y_test))
            else:
                print("pickle file : " + model + '_test_mrd_' + str(n_polarity) + '.pkl')
                pickle_file(model + '_test_mrd_' + str(n_polarity) + '.pkl', (X_test, y_test))

        print("Fitting SVM")
        clf = svm.SVC(kernel='linear', C=1, gamma=0.1)
        clf.fit(X_train, y_train)

        if hashing_trick:
            print('Saving model : ' + model + '_train_mrd_' + str(n_polarity) + '_hash.pkl')
            pickle_file(model + '_train_mrd_' + str(n_polarity) + '_hash.pkl', clf)
        else:
            if negation:
                print('Saving model : ' + model + '_train_mrd_' + str(n_polarity) + '_n.pkl')
                pickle_file(model + '_train_mrd_' + str(n_polarity) + '_n.pkl', clf)
            else:
                print('Saving model : ' + model + '_train_mrd_' + str(n_polarity) + '.pkl')
                pickle_file(model + '_train_mrd_' + str(n_polarity) + '.pkl', clf)
    else:
        parser.error("-d, --dataset requires imdb or mrd")
