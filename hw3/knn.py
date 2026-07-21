import numpy as np
from statistics import mode
from abc import abstractmethod, ABC
from data import StandardScaler


class KNN(ABC):
    def __init__(self, k):
        """ object instantiation, save k and define a scaler object """
        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """ fit scaler and save X_train and y_train """
        self.scaler.fit(X_train)
        self.X_train = self.scaler.transform(X_train)
        self.y_train = np.array(y_train)
        #self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        pass

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set """
        distances = []
        for row in self.X_train:
            distances.append(self.dist(x, row))
        distances = np.array(distances)
        k_closest_indices = np.argsort(distances)[:self.k]
        return k_closest_indices

    @staticmethod
    def dist(x1, x2):
        """returns Euclidean distance between x1 and x2"""
        return np.sqrt(np.sum((x1 - x2) ** 2))


class ClassificationKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        X_test_transformed = self.scaler.transform(X_test)
        labels = []
        for x in X_test_transformed:
            labels.append(mode(self.y_train[self.neighbours_indices(x)]))
        return np.array(labels)


class RegressionKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        X_test_transformed = self.scaler.transform(X_test)
        labels = []
        for x in X_test_transformed:
            labels.append(np.mean(self.y_train[self.neighbours_indices(x)]))
        return np.array(labels)
