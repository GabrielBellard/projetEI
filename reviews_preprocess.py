# coding=utf-8
# encoding: utf-8

from __future__ import print_function
import string

import scipy.sparse
from bs4 import BeautifulSoup

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
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
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

np.random.seed(1337)  # for reproducibility

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


def build_dict_feature_imdb(double_features):
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

    if model == "svm":
        X_train, vectorizer_fitted = build_dic_svm(sentences_train, double_features)
        X_test, _ = build_dic_svm(sentences_test, double_features, vectorizer_fitted)
        n = X_train.shape[0] / 2
        y_train = [1] * n + [0] * n
        y_test = [1] * n + [0] * n

    elif model == "cnn" or model == "lstm":
        X_train, tokenizer = build_dic_nn(sentences=sentences_train, double_features=double_features)

        X_test, _ = build_dic_nn(sentences=sentences_test, double_features=double_features, tokenizer=tokenizer)

        n = len(X_train) / 2
        y_train = [1] * n + [0] * n
        y_test = [1] * n + [0] * n

    if feature_selection:
        print("Doing feature selection")
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


def build_dict_feature_spd(double_features):
    sentences_pos = []

    ff = os.path.join(dataset_path_spd, 'rt-polarity_utf8.pos')

    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences pos"):
            # time.sleep(0.001)
            sentences_pos.append(line)

    sentences_neg = []
    ff = os.path.join(dataset_path_spd, 'rt-polarity_utf8.neg')
    with io.open(ff, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, desc="sentences neg"):
            # time.sleep(0.001)
            sentences_neg.append(line)

    sentences = sentences_pos + sentences_neg

    y = [1] * (len(sentences_pos)) + [0] * (len(sentences_neg))

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=58)

    if model == "svm":
        X_train, vectorizer = build_dic_svm(sentences_train, double_features)
        X_test, _ = build_dic_svm(sentences_test, double_features, vectorizer)
    elif model == "cnn" or model == "lstm":
        X_train, tokenizer = build_dic_nn(sentences=sentences_train, double_features=double_features)
        X_test, _ = build_dic_nn(sentences=sentences_test, double_features=double_features, tokenizer=tokenizer)

    if feature_selection:
        print("Doing feature selection")
        if hashing_trick:
            fselect = SelectKBest(chi2, k=9500)
        else:
            if negation:
                fselect = SelectKBest(chi2, k=9500)
            else:
                fselect = SelectKBest(chi2, k=8500)

        X_train = fselect.fit_transform(X_train, y_train)

        X_test = fselect.transform(X_test)

    return X_train, X_test, y_train, y_test


def build_dic_svm(sentences, double_features, vectorizer_fitted=None):
    do_negation = negation

    # if training data because we do not fit testing data
    if hashing_trick and vectorizer_fitted is None:
        hasher = HashingVectorizer(norm=None, non_negative=True)

        vectorizer = Pipeline([('hasher', hasher), ('tf_idf', TfidfTransformer())])

    elif not hashing_trick and vectorizer_fitted is None:
        # some parameters that we tuned
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 6),
                                     sublinear_tf=True)

    elif vectorizer_fitted is not None:
        vectorizer = vectorizer_fitted

    if sentiwordnet:
        X_polarity = get_polarity(double_features, sentences)

    sentences_stem = Parallel(n_jobs=num_cores)(delayed(preprocessing_sentences)(sentence, do_negation) for
                                                sentence in tqdm.tqdm(sentences, desc="preprocessing"))

    sentences_stem2 = [' '.join(term) for term in sentences_stem]

    start = time.time()
    if vectorizer_fitted is not None:
        X_sentences = vectorizer.transform(sentences_stem2)
    else:
        X_sentences = vectorizer.fit_transform(sentences_stem2)
    end = time.time()
    elapsed = end - start
    print("temps de vectorisation : " + str(elapsed))

    if sentiwordnet:
        # if sentiwordnet we add a feature
        X = scipy.sparse.hstack([X_sentences, X_polarity])
        return X, vectorizer

    else:
        return X_sentences, vectorizer


def build_dic_nn(sentences, double_features, tokenizer=None):
    do_negation = negation

    if sentiwordnet:
        X_polarity = get_polarity(double_features, sentences)

    sentences_stem = Parallel(n_jobs=num_cores)(delayed(preprocessing_sentences)(sentence, do_negation) for
                                                sentence in tqdm.tqdm(sentences, desc="preprocessing"))

    sentences_stem2 = [' '.join(term) for term in sentences_stem]

    X_sentences, tokenizer = build_w2c_dic(sentences_stem2, tokenizer)

    if sentiwordnet:
        X = scipy.sparse.hstack([X_sentences, X_polarity])
        return X, tokenizer

    else:
        return X_sentences, tokenizer


def build_w2c_dic(sentences_stem2, tokenizer=None):

    from keras.preprocessing.text import Tokenizer

    sentences_stem2 = [s.encode('utf-8') for s in sentences_stem2]

    if tokenizer is None:
        tokenizer = Tokenizer(nb_words=50)
        tokenizer.fit_on_texts(sentences_stem2)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        pickle_file("word_index_"+model+"_"+corpus, word_index)
    else:
        tokenizer = tokenizer

    X_sentences = tokenizer.texts_to_sequences(sentences_stem2)

    return X_sentences, tokenizer


# def train_w2v(sentences_stem2, vocab_dim, n_exposures):
#
#     embeddings_index = {}
#     f = open('glove.6B.100d.txt')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#
#     print('Found %s word vectors.' % len(embeddings_index))
#
#     embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#     for word, i in word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#
#     pickle_file("cnn_embedding_weights_" + str(corpus) + "n_" + str(negation) + ".pkl", embedding_weights)
#
#     return w2indx


def get_polarity(double_features, sentences):
    polarity_arr2 = []
    polarity_arr = Parallel(n_jobs=num_cores)(delayed(compute_polarity)(sentence, double_features) for
                                              sentence in tqdm.tqdm(sentences, desc="compute polarity"))
    [polarity_arr2.append(dict_) for list in polarity_arr for dict_ in list]
    feature_hasher = FeatureHasher(non_negative=True)
    X_polarity = feature_hasher.fit_transform(polarity_arr2)
    return X_polarity


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
    # Remove punctuation, stopword, html tag and then stemmering

    punctuation = set(string.punctuation)
    stemmer = nltk.PorterStemmer()
    sentence = BeautifulSoup(sentence, "lxml").get_text()

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


def construct_cnn(vocab_dim, maxlen, batch_size, X_train, y_train, embedding_matrix):
    clf = Sequential()


    n_symbols = len(embedding_matrix)

    print(len(embedding_matrix))

    clf.add(Embedding(input_dim=n_symbols,
                      output_dim=vocab_dim,
                      input_length=maxlen,
                      weights=[embedding_matrix],
                      dropout=0.2))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    clf.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=filter_length,
                          border_mode='valid',
                          activation='relu',
                          subsample_length=1))
    # we use max pooling:
    clf.add(GlobalMaxPooling1D())
    # We add a vanilla hidden layer:
    clf.add(Dense(hidden_dims))
    clf.add(Dropout(0.2))
    clf.add(Activation('relu'))
    # We project onto a single unit output layer, and squash it with a sigmoid:
    clf.add(Dense(1))
    clf.add(Activation('sigmoid'))

    clf.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    clf.fit(X_train, y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            validation_split=0.1)

    return clf


def construct_lstm(vocab_dim, maxlen, batch_size, X_train, y_train, embedding_matrix):

    clf = Sequential()  # or Graph or whatever
    n_symbols = len(embedding_matrix)
    clf.add(Embedding(input_dim=n_symbols,
                      output_dim=vocab_dim,
                      mask_zero=True,
                      weights=[embedding_matrix],
                      input_length=maxlen))  # Adding Input Length
    clf.add(LSTM(vocab_dim))
    # clf.add(Dropout(0.5))
    clf.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(1, activation='relu'))

    print('Compiling the Model...')
    clf.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1)

    return clf


def construct_nn(vocab_dim, maxlen, batch_size, nn, X_train, X_test, y_train, y_test):

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Loading model...')

    embeddings_index = {}
    f = open(os.path.join('glove/','glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    word_index = unpickle_file("word_index_"+model+"_"+corpus)
    embedding_matrix = np.zeros((len(word_index) + 1, vocab_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    if nn == "cnn":
        clf = construct_cnn(vocab_dim=vocab_dim, maxlen=maxlen, batch_size=batch_size,
                            X_train=X_train, y_train=y_train, embedding_matrix=embedding_matrix)
    elif nn == "lstm":
        clf = construct_lstm(vocab_dim=vocab_dim, maxlen=maxlen, batch_size=batch_size,
                             X_train=X_train, y_train=y_train, embedding_matrix=embedding_matrix)

    return clf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model you want to use')
    parser.add_argument('-d', '--dataset', help='which dataset imdb or spd', required=True, nargs=1, type=str)
    parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet and precise number of features', required=False,
                        type=int)
    parser.add_argument('-m', '--model', help='which model cnn, lstm, svm', required=True, nargs=1, type=str)
    parser.add_argument('-j', '--joblib', help='number of jobs for parallelism', required=False, type=int,
                        default=num_cores)
    parser.add_argument('-hash', '--hash', help='whether use hashing trick', required=False, action='store_true')
    parser.add_argument('-n', '--negation', help='whether do negation(bad result)', required=False, action='store_true')
    parser.add_argument('-f', '--feature', help='whether use feature selection', required=False, action='store_true')
    args = parser.parse_args()

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
    elif "svm" in args.model:
        model = "svm"
    elif "cnn" in args.model:
        model = "cnn"

    if args.hash:
        hashing_trick = True

    if args.negation:
        negation = True

    if args.feature:
        feature_selection = True

    corpus = str(args.dataset)

    if "imdb" in corpus:
        X_train, X_test, y_train, y_test = build_dict_feature_imdb(double_features)

    elif "spd" in corpus:
        X_train, X_test, y_train, y_test = build_dict_feature_spd(double_features)

    else:
        parser.error("-d, --dataset requires imdb or spd")

    if model == "svm":

        print("Fitting SVM")
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)

        print("pickle file : {0}_train_{1}_{2}_h{3}_n{4}_f{5}.pkl".format(model, corpus, str(n_polarity),
                                                                          str(hashing_trick), str(negation),
                                                                          str(feature_selection)))

        pickle_file('{0}_train_{1}_{2}_h{3}_n{4}_f{5}.pkl'.format(model, corpus, str(n_polarity),
                                                                  str(hashing_trick), str(negation),
                                                                  str(feature_selection)),clf)

    elif model == "cnn" or model == "lstm":
        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, LSTM
        from keras.layers import Embedding
        from keras.layers import Convolution1D, GlobalMaxPooling1D

        if "spd" in corpus:
            clf = construct_nn(vocab_dim=vocab_dim_SPD, maxlen=maxlen_SPD, batch_size=batch_size_SPD, nn=model,
                               X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        elif "imdb" in corpus:
            clf = construct_nn(vocab_dim=vocab_dim_IMDB, maxlen=maxlen_IMDB, batch_size=batch_size_IMDB, nn=model,
                               X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        print("saving file : {0}_train_{1}_{2}_h{3}_n{4}.h5".format(model, corpus, str(n_polarity),
                                                                    str(hashing_trick), str(negation)))
        clf.save(directory+'{0}_train_{1}_{2}_h{3}_n{4}.h5'.format(model, corpus, str(n_polarity),
                                                         str(hashing_trick), str(negation)))

    print("pickle file : {0}_test_{1}_{2}_h{3}_n{4}_f{5}.pkl".format(model, corpus, str(n_polarity),
                                                                     str(hashing_trick), str(negation),
                                                                     str(feature_selection)))

    pickle_file('{0}_test_{1}_{2}_h{3}_n{4}_f{5}.pkl'.format(model, corpus, str(n_polarity),
                                                             str(hashing_trick), str(negation), str(feature_selection)),
                                                            (X_test, y_test))
