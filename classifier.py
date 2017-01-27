# coding=utf-8
import argparse

from sklearn.metrics import classification_report, accuracy_score
from utils import *
import time

parser = argparse.ArgumentParser(description='Test the model you want to test')
parser.add_argument('-d', '--dataset', help='which dataset imdb or mrd', required=True, nargs=1, type=str)
parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet', action='store_true')
parser.add_argument('-np', '--number_polarity', help='if 2 features pos and neg or global', required=False, type=int,
                    nargs=1)
parser.add_argument('-m', '--model', help='which model svm or lstm', required=True, nargs=1, type=str)
args = parser.parse_args()

n_polarity = 0

if args.sentiwordnet is True and args.number_polarity is None:
    parser.error("-s, --sentiwordnet requires --np 1|2 ")

if args.sentiwordnet is False:
    n_polarity = 0

if args.sentiwordnet and 2 in args.number_polarity :
    double_features = True
    n_polarity = 2

elif args.sentiwordnet and 1 in args.number_polarity:
    double_features = False
    n_polarity = 1

if "lstm" in args.model:
    model = "lstm"
else:
    model = "svm"

if "imdb" in str(args.dataset):

    print('Loading'+model+'_train_imdb_' + str(n_polarity) + '.pkl')
    clf = unpickle_file(model+'_train_imdb_' + str(n_polarity) + '.pkl')
    print('Loading' +model+'_test_imdb_' + str(n_polarity) + '.pkl')
    X_test, y_test = unpickle_file(model+'_test_imdb_' + str(n_polarity) + '.pkl')

elif "mrd" in str(args.dataset):

    print('Loading '+model+'_train_mrd_' + str(n_polarity) + '.pkl')
    clf = unpickle_file('train_mrd_' + str(n_polarity) + '.pkl')
    print('Loading'+model+'_test_mrd_' + str(n_polarity) + '.pkl')
    X_test, y_test = unpickle_file(model+'_test_mrd_' + str(n_polarity) + '.pkl')

else:
    parser.error("-d, --dataset requires imdb or mrd")



y_pred = clf.predict(X_test)


print("Results for SVC(kernel=linear)")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

