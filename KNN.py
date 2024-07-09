import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from model import Model

class KNN(Model):
    data = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    bestK = None

    def setupData(self, filePath):
        self.data = pd.read_csv(filePath)
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

    def getBestK(self):
        knn = neighbors.KNeighborsClassifier(p=1, metric='minkowski', weights='distance')
        param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 33, 35, 37, 40, 42, 45, 47, 50]}
        grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
        grid_search_knn.fit(self.X_train, self.y_train)
        best_k_knn = grid_search_knn.best_params_['n_neighbors']
        print("Best number of neighbors (K) for KNN:", best_k_knn)

        self.bestK = best_k_knn

    def trainModel(self):
        knn = neighbors.KNeighborsClassifier(n_neighbors=self.bestK, p=1, metric='minkowski', weights='distance')

        knn.fit(self.X_train, self.y_train)

        self.model = knn

        y_pred_knn = knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        print("The KNN accuracy score without scaling is:", accuracy_knn)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        knn.fit(X_train_scaled, self.y_train)

        y_pred_knn_scaled = knn.predict(self.X_test)
        accuracy_knn_scaled = accuracy_score(self.y_test, y_pred_knn_scaled)
        print("The KNN accuracy score on scaled data is:", accuracy_knn_scaled)

    def plotModel(self, feature1, feature2):
        features = [feature1, feature2]
        knn = neighbors.KNeighborsClassifier(n_neighbors=self.bestK, p=1, metric='minkowski', weights='distance')
        knn.fit(self.X_train[features], self.y_train)

        h = 0.04
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        x_min, x_max = self.X_train[features].iloc[:, 0].min() - 1, self.X_train[features].iloc[:, 0].max() + 1
        y_min, y_max = self.X_train[features].iloc[:, 1].min() - 1, self.X_train[features].iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        plt.scatter(self.X_train[features].iloc[:, 0], self.X_train[features].iloc[:, 1], c=self.y_train, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.show()

        y_pred = knn.predict(self.X_test[features])
        print('Prediction Accuracy is %f' % accuracy_score(y_pred, self.y_test))

if __name__ == '__main__':
    thing = KNN()
    thing.setupData("matchedData/AllYears.csv")
    # thing.setupData("matchedData/2019-20.csv")
    # thing.setupData("matchedData/2020-21.csv")
    # thing.setupData("matchedData/2021-22.csv")
    # thing.setupData("matchedData/2022-23.csv")
    # thing.setupData("matchedData/2023-24.csv")
    thing.getBestK()
    thing.trainModel()
    thing.saveModel("models/KNN.sav")
    thing.loadModel("models/KNN.sav")
    # thing.plotModel('HOME PTS', 'AWAY PTS')
    # thing.plotModel('HOME FG%', 'AWAY FG%')