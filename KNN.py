import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class KNN:
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

        y_pred_knn = knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        print("The KNN accuracy score without scaling is:", accuracy_knn)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        knn.fit(X_train_scaled, self.y_train)

        y_pred_knn_unscaled = knn.predict(self.X_test)
        accuracy_knn_unscaled = accuracy_score(self.y_test, y_pred_knn_unscaled)
        print("The KNN accuracy score on unscaled data is:", accuracy_knn_unscaled)
    

if __name__ == '__main__':
    thing = KNN()
    thing.setupData()
    thing.getBestK()
    thing.trainModel()