# coding=utf-8
import argparse

from sklearn.metrics import classification_report, accuracy_score, auc, roc_auc_score
from utils import *
from constants import *
import time

parser = argparse.ArgumentParser(description='Test the model you want to test')
parser.add_argument('-d', '--dataset', help='which dataset imdb or spd', required=True, nargs=1, type=str)
parser.add_argument('-s', '--sentiwordnet', help='use sentiwordnet', type=int, required=False)
parser.add_argument('-m', '--model', help='which model svm or lstm', required=True, nargs=1, type=str)
parser.add_argument('-hash', '--hash', help='whether use hashing trick', required=False, action='store_true')
parser.add_argument('-n', '--negation', help='whether do negation(bad result)', required=False, action='store_true')
parser.add_argument('-f', '--feature', help='whether use feature selection', required=False, action='store_true')
args = parser.parse_args()

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

if model == "lstm" or model == "cnn":
    from keras.models import load_model

    print("Loading file : {0}_train_{1}_{2}_h{3}_n{4}.h5".format(model, str(args.dataset), str(n_polarity),
                                                                 str(hashing_trick), str(negation)))
    clf = load_model('{0}_train_{1}_{2}_h{3}_n{4}.h5'.format(model, str(args.dataset), str(n_polarity),
                                                              str(hashing_trick), str(negation)))
else:
    print("Loading file : {0}_train_{1}_{2}_h{3}_n{4}_f{5}.pkl".format(model, str(args.dataset), str(n_polarity),
                                                                 str(hashing_trick), str(negation), str(feature_selection)))
    clf = unpickle_file('{0}_train_{1}_{2}_h{3}_n{4}_f{5}.pkl'.format(model, str(args.dataset), str(n_polarity),
                                                              str(hashing_trick), str(negation), str(feature_selection)))

print("Loading file : {0}_test_{1}_{2}_h{3}_n{4}_f{5}.pkl".format(model, str(args.dataset), str(n_polarity),
                                                            str(hashing_trick), str(negation), str(feature_selection)))

X_test, y_test = unpickle_file('{0}_test_{1}_{2}_h{3}_n{4}_f{5}.pkl'.format(model, str(args.dataset), str(n_polarity),
                                                    str(hashing_trick), str(negation), str(feature_selection)))

print("Prédiction en cours...")
y_pred = clf.predict(X_test)

print("Results for SVC(kernel=linear)")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
