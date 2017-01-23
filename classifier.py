# coding=utf-8
import argparse

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from utils import *
import time

parser = argparse.ArgumentParser(description='Test the model you want to test')
parser.add_argument('-d', '--dataset', help='which dataset imdb or mrd', required=True, nargs=1, type=str)
parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet', action='store_true')
parser.add_argument('-np', '--number_polarity', help='if 2 features pos and neg or global', required=False, type=int,
                    nargs=1)
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


if "imdb" in str(args.dataset):

    print('Loading train_imdb_' + str(n_polarity) + '.pkl')
    X_train, y_train = unpickle_file('train_imdb_' + str(n_polarity) + '.pkl')
    print('Loading test_imdb_' + str(n_polarity) + '.pkl')
    X_test, y_test = unpickle_file('test_imdb_' + str(n_polarity) + '.pkl')

elif "mrd" in str(args.dataset):

    print('Loading train_mrd_' + str(n_polarity) + '.pkl')
    X_train, y_train = unpickle_file('train_mrd_' + str(n_polarity) + '.pkl')
    print('Loading test_mrd_' + str(n_polarity) + '.pkl')
    X_test, y_test = unpickle_file('test_mrd_' + str(n_polarity) + '.pkl')

else:
    parser.error("-d, --dataset requires imdb or mrd")

print('Fitting SVC Linéaire')

classifier_linear = svm.SVC(kernel='linear', C=1)
t0 = time.time()
classifier_linear.fit(X_train, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(X_test)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(y_test, prediction_linear))
print(accuracy_score(y_test, prediction_linear))

print('Fitting SVC LibLinéaire')
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(X_train, y_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(X_test)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1

print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(y_test, prediction_liblinear))
print(accuracy_score(y_test, prediction_liblinear))
