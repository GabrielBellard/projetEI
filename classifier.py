from sklearn import svm
from sklearn.metrics import classification_report
from utils import *
import time

X_train, y_train = unpickle_file("train.pkl")
X_test, y_test = unpickle_file("test.pkl")

print "Classifier RBF"

classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(X_train, y_train)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(X_test)
t2 = time.time()
time_rbf_train = t1 - t0
time_rbf_predict = t2 - t1

print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print("Results for SVC(kernel=rbf)")
print(classification_report(y_test, prediction_rbf))

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

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(X_train, y_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(X_test)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1

# Print results in a nice table


print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(y_test, prediction_liblinear))