import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SVM:
    data = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    bestK = None

    def setupData(self):
        self.data = pd.read_csv("matchedData/AllYears.csv")

        pd.set_option('display.max_columns', None)


        self.data['Winner (H/A)'] = self.data['Winner (H/A)'].replace({'H': 1, 'A': 0})

        self.data.drop(['DATE', 'HOME TEAM', 'AWAY TEAM'], axis=1, inplace=True)

        self.data = self.data.astype(float)

        X = self.data.drop("Winner (H/A)", axis=1)
        Y = self.data["Winner (H/A)"]

        X = np.array(X)
        Y = np.array(Y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1)

        print("Train set shape:", self.X_train.shape)
        print("Test set shape:", self.X_test.shape)

    def trainModel(self):
        svm_linear = svm.LinearSVC(C=1.0).fit(self.X_train, self.y_train)
        svm_linear_predictions = svm_linear.predict(self.X_test)
        svm_linear_accuracy = accuracy_score(self.y_test, svm_linear_predictions)
        print("Linear SVM Accuracy:", svm_linear_accuracy)

        svm_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(self.X_train, self.y_train)
        svm_rbf_predictions = svm_rbf.predict(self.X_test)
        svm_rbf_accuracy = accuracy_score(self.y_test, svm_rbf_predictions)
        print("RBF SVM Accuracy:", svm_rbf_accuracy)

        svm_poly = svm.SVC(kernel='poly', degree=3, C=1.0).fit(self.X_train, self.y_train)
        svm_poly_predictions = svm_poly.predict(self.X_test)
        svm_poly_accuracy = accuracy_score(self.y_test, svm_poly_predictions)
        print("Polynomial SVM Accuracy:", svm_poly_accuracy)

if __name__ == '__main__':
    thing = SVM()
    thing.setupData()
    thing.trainModel()