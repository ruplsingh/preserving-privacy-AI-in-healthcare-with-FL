from sklearn import metrics
from utils import get_test_train_data
from sklearn import svm

x_train, x_test, y_train, y_test = get_test_train_data()

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print("SVM Accuracy           : ", metrics.accuracy_score(y_test, y_predict) * 100, "%")
