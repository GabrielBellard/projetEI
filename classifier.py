from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from utils import *
import time

import xgboost

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

X_train, y_train = unpickle_file("train_mrd_sentiword.pkl")
X_test, y_test = unpickle_file("test_mrd_sentiword.pkl")

# print "Classifier RBF"
#
# classifier_rbf = svm.SVC()
# t0 = time.time()
# classifier_rbf.fit(X_train, y_train)
# t1 = time.time()
# prediction_rbf = classifier_rbf.predict(X_test)
# t2 = time.time()
# time_rbf_train = t1 - t0
# time_rbf_predict = t2 - t1
#
# print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
# print("Results for SVC(kernel=rbf)")
# print(classification_report(y_test, prediction_rbf))
#
# Perform classification with SVM, kernel=linear

classifier_linear = svm.SVC(kernel='linear')
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
# #
# # Perform classification with SVM, kernel=linear
# classifier_liblinear = svm.LinearSVC()
# t0 = time.time()
# classifier_liblinear.fit(X_train, y_train)
# t1 = time.time()
# prediction_liblinear = classifier_liblinear.predict(X_test)
# t2 = time.time()
# time_liblinear_train = t1 - t0
# time_liblinear_predict = t2 - t1

# Print results in a nice table


# print("Results for LinearSVC()")
# print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
# print(classification_report(y_test, prediction_liblinear))
# print(accuracy_score(y_test, prediction_liblinear))
#
# print 'Test XGBOOST'
#
# clf = xgboost.XGBClassifier(learning_rate =0.1,
#                              n_estimators=1000,
#                              max_depth=9,
#                              min_child_weight=3,
#                              gamma=0,
#                              subsample=0.8,
#                              colsample_bytree=0.8,
#                              objective= 'binary:logistic',
#                              nthread=4,
#                              scale_pos_weight=1,
#                              seed=27)
# t0 = time.time()
# clf.fit(X_train, y_train)
# t1 = time.time()
# xgboost_predict = clf.predict(X_test)
# t2 = time.time()
# time_xgboost_train = t1 - t0
# time_xgboost_predict = t2 - t1
#
# print("Results for xgboost")
# print("Training time: %fs; Prediction time: %fs" % (time_xgboost_train, time_xgboost_predict))
# print(classification_report(y_test, xgboost_predict))
# print(accuracy_score(y_test, xgboost_predict))

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
#
# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
#
# gsearch1 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
#  min_child_weight=3, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
#  param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(X_train, y_train)
#
# print gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# print 'Test DT'
#
# clf = DecisionTreeClassifier()
# t0 = time.time()
# clf.fit(X_train, y_train)
# t1 = time.time()
# dt_predict = clf.predict(X_test)
# t2 = time.time()
# time_xgboost_train = t1 - t0
# time_xgboost_predict = t2 - t1
#
# print("Results for dt")
# print("Training time: %fs; Prediction time: %fs" % (time_xgboost_train, time_xgboost_predict))
# print(classification_report(y_test, dt_predict))
# print(accuracy_score(y_test, dt_predict))
