from utils import get_test_train_data
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = get_test_train_data()

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, y_train)

print("KNN Accuracy           : ", knn.score(x_test, y_test) * 100, "%")
