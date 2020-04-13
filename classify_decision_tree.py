from utils import get_test_train_data
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = get_test_train_data()

depression_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
depression_classifier.fit(x_train, y_train)

y_predicted = depression_classifier.predict(x_test)
print("Decision Tree Accuracy : ", accuracy_score(y_test, y_predicted) * 100, "%")
