# coding=utf-8
from __future__ import print_function
from sklearn import svm
from reviews_preprocess import *
from utils import *
import time


import metacritic
import pprint

if __name__ == '__main__':
    clf = unpickle_file("svm_train_['spd']_0_hFalse_nFalse_fFalse.pkl")

    critics = metacritic.get_movie_critics_for_letter("jack-and-jill")

    X_test = build_dic_svm(critics, double_features, True)

    prediction_linear = clf.predict(X_test)

    prediction_linear = prediction_linear.tolist()

    pos = float(prediction_linear.count(1))
    neg = float(prediction_linear.count(0))

    len = float(len(prediction_linear))

    ratio_pos = (pos / len) * 100
    ration_neg = (neg / len) * 100

    if pos > neg:
        print("Pour le theme : " + str("titanic") + ", l'opinion des utilisateurs de twitter est plutot positif (a ", int(ratio_pos), " %)")
    else:
        print("Pour le theme : " + str("titanic") + ", l'opinion des utilisateurs de twitter est plutot negatif (a ", int(ration_neg), " %)")

    from pylab import *

    # make a square figure and axes
    figure(1, figsize=(6, 6))
    ax = axes([0.1, 0.1, 0.8, 0.8])

    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Positif', 'Negatif'

    fracs = [ratio_pos, ration_neg]
    explode = (0, 0.05)

    pie(fracs, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90)

    title('Opinion ' + str("titanic"), bbox={'facecolor': '0.8', 'pad': 5})

    show()
