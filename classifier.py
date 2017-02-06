# coding=utf-8
import argparse

from sklearn.metrics import classification_report, accuracy_score, auc, roc_auc_score
from utils import *
import time

parser = argparse.ArgumentParser(description='Test the model you want to test')
parser.add_argument('-d', '--dataset', help='which dataset imdb or mrd', required=True, nargs=1, type=str)
parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet', type=int, required=False)
parser.add_argument('-m', '--model', help='which model svm or lstm', required=True, nargs=1, type=str)
parser.add_argument('-hash', '--hash', help='whether use hashing trick', required=False, action='store_true')
parser.add_argument('-n', '--negation', help='whether do negation(bad result)', required=False, action='store_true')
args = parser.parse_args()

n_polarity = 0
hashing_trick = False
negation = False

if args.sentiwordnet is True and args.number_polarity is None:
    parser.error("-s, --sentiwordnet requires --np 1|2 ")


if args.sentiwordnet == 2:
    double_features = True
    n_polarity = 2

elif args.sentiwordnet == 1:
    double_features = False
    n_polarity = 1
else:
    n_polarity = 0
    double_features = False

if "lstm" in args.model:
    model = "lstm"
else:
    model = "svm"

if args.hash:
    hashing_trick = True

if args.negation:
    negation = True

if "imdb" in str(args.dataset):

    if hashing_trick:
        print('Loading '+model+'_train_imdb_' + str(n_polarity) + '_hash.pkl')
        clf = unpickle_file(model+'_train_imdb_' + str(n_polarity) + '_hash.pkl')
        print('Loading ' +model+'_test_imdb_' + str(n_polarity) + '_hash.pkl')
        X_test, y_test = unpickle_file(model+'_test_imdb_' + str(n_polarity) + '_hash.pkl')
    else:
        if negation:
            print('Loading ' + model + '_train_imdb_' + str(n_polarity) + '_n.pkl')
            clf = unpickle_file(model + '_train_imdb_' + str(n_polarity) + '_n.pkl')
            print('Loading ' + model + '_test_imdb_' + str(n_polarity) + '_n.pkl')
            X_test, y_test = unpickle_file(model + '_test_imdb_' + str(n_polarity) + '_n.pkl')
        else:
            print('Loading ' + model + '_train_imdb_' + str(n_polarity) + '.pkl')
            clf = unpickle_file(model + '_train_imdb_' + str(n_polarity) + '.pkl')
            print('Loading ' + model + '_test_imdb_' + str(n_polarity) + '.pkl')
            X_test, y_test = unpickle_file(model + '_test_imdb_' + str(n_polarity) + '.pkl')

elif "mrd" in str(args.dataset):
    if hashing_trick:
        print('Loading '+model+'_train_mrd_' + str(n_polarity) + '_hash.pkl')
        clf = unpickle_file(model+'_train_mrd_' + str(n_polarity) + '_hash.pkl')
        print('Loading '+model+'_test_mrd_' + str(n_polarity) + '_hash.pkl')
        X_test, y_test = unpickle_file(model+'_test_mrd_' + str(n_polarity) + '_hash.pkl')

    else:
        if negation:
            print('Loading ' + model + '_train_mrd_' + str(n_polarity) + '_n.pkl')
            clf = unpickle_file(model + '_train_mrd_' + str(n_polarity) + '_n.pkl')
            print('Loading ' + model + '_test_mrd_' + str(n_polarity) + '_n.pkl')
            X_test, y_test = unpickle_file(model + '_test_mrd_' + str(n_polarity) + '_n.pkl')
        else:
            print('Loading ' + model + '_train_mrd_' + str(n_polarity) + '.pkl')
            clf = unpickle_file(model + '_train_mrd_' + str(n_polarity) + '.pkl')
            print('Loading ' + model + '_test_mrd_' + str(n_polarity) + '.pkl')
            X_test, y_test = unpickle_file(model + '_test_mrd_' + str(n_polarity) + '.pkl')

else:
    parser.error("-d, --dataset requires imdb or mrd")


print("Prédiction en cours...")
y_pred = clf.predict(X_test)


print("Results for SVC(kernel=linear)")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# print(auc(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))

