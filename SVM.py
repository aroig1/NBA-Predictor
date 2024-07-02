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
    svm_linear = None
    svm_rbf = None
    svm_poly = None
    

    def setupData(self):
        self.data = pd.read_csv("matchedData/AllYears.csv")
        pd.set_option('display.max_columns', None)
        self.data['Winner (H/A)'] = self.data['Winner (H/A)'].replace({'H': 1, 'A': 0})
        self.data.drop(['DATE', 'HOME TEAM', 'AWAY TEAM'], axis=1, inplace=True)
        self.data = self.data.astype(float)

        # X = self.data.drop("Winner (H/A)", axis=1)
        # Y = self.data["Winner (H/A)"]
        # X = np.array(X)
        # Y = np.array(Y)
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1)

        train, test = train_test_split(self.data, test_size=0.1, random_state = 0)
        self.X_train = train.drop("Winner (H/A)", axis=1)
        self.y_train = train['Winner (H/A)']
        self.X_test = test.drop("Winner (H/A)", axis=1)
        self.y_test = test['Winner (H/A)']

        print("Train set shape:", self.X_train.shape)
        print("Test set shape:", self.X_test.shape)

    def trainModel(self):
        self.svm_linear = svm.LinearSVC(C=1.0).fit(self.X_train, self.y_train)
        svm_linear_predictions = self.svm_linear.predict(self.X_test)
        svm_linear_accuracy = accuracy_score(self.y_test, svm_linear_predictions)
        print("Linear SVM Accuracy:", svm_linear_accuracy)

        self.svm_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(self.X_train, self.y_train)
        svm_rbf_predictions = self.svm_rbf.predict(self.X_test)
        svm_rbf_accuracy = accuracy_score(self.y_test, svm_rbf_predictions)
        print("RBF SVM Accuracy:", svm_rbf_accuracy)

        self.svm_poly = svm.SVC(kernel='poly', degree=3, C=1.0).fit(self.X_train, self.y_train)
        svm_poly_predictions = self.svm_poly.predict(self.X_test)
        svm_poly_accuracy = accuracy_score(self.y_test, svm_poly_predictions)
        print("Polynomial SVM Accuracy:", svm_poly_accuracy)

    def plotModel(self):
        titles = ['SVC with rbf kernal', 'SVC with polynomial (degree 3) kernel']
        h = 0.02
        # create a mesh to plot in
        x_min, x_max = self.X_train.iloc[:, 0].min() - 1, self.X_train.iloc[:, 0].max() + 1
        y_min, y_max = self.X_train.iloc[:, 1].min() - 1, self.X_train.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        plt.figure(figsize = (12, 10))
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
        for i, model in enumerate((self.svm_rbf, self.svm_poly)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=cmap_bold, alpha=0.3)

            # Plot also the training points
            plt.scatter(self.X_train.iloc[:, 0], self.X_train.iloc[:, 1], c=self.y_train, cmap=cmap_bold, edgecolor = 'k', s=20)
            plt.xlabel('O_or%')
            plt.ylabel('D_3p')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

        plt.show()

if __name__ == '__main__':
    thing = SVM()
    thing.setupData()
    thing.trainModel()
    # thing.plotModel() # NOT WORKING