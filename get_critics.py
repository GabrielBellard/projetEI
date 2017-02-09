# coding=utf-8
from __future__ import print_function
from sklearn import svm
from reviews_preprocess import *
from utils import *
import time


import metacritic
import pprint


X_train, y_train = unpickle_file("train_mrd_sentiword_postag.pkl")

critics = metacritic.get_movie_critics_for_letter("jack-and-jill")

pprint.pprint(critics)

X_test = build_dic_svm(critics, double_features)

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
