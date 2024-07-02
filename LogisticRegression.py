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
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class LogisticRegression:
    data = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    bestC = 0.1

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.15)

        print("Train set shape:", self.X_train.shape)
        print("Test set shape:", self.X_test.shape)

    def getBestC(self):
        model = linear_model.LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear')

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        bestC = grid_search.best_params_['C']
        print("Best C value:", bestC)

        return bestC
    
    def TBT(self):
        logreg = linear_model.LogisticRegression(C=self.bestC, max_iter=1000)
        logreg.fit(self.X_train, self.y_train)

        y_pred = logreg.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("The logistic regression accuracy score is:", accuracy)

        logreg_regularized = linear_model.LogisticRegression(penalty='l1', C=self.bestC, max_iter=1000, solver='liblinear')
        logreg_regularized.fit(self.X_train, self.y_train)

        y_pred_regularized = logreg_regularized.predict(self.X_test)
        accuracy_regularized = accuracy_score(self.y_test, y_pred_regularized)
        print("The regularized logistic regression accuracy score is:", accuracy_regularized)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        logreg_regularized.fit(X_train_scaled, self.y_train)

        y_pred_unscaled = logreg_regularized.predict(self.X_test)
        accuracy_unscaled = accuracy_score(self.y_test, y_pred_unscaled)
        print("The logistic regression accuracy score on unscaled data is:", accuracy_unscaled)

    def correlation_heatmap(self):
        _ , ax = plt.subplots(figsize =(20, 15))
        colormap = sns.diverging_palette(220, 10, as_cmap = True)
        sns.heatmap(
            self.data.corr(),
            cmap = colormap,
            square=True,
            cbar_kws={'shrink':.9 },
            ax=ax,
            annot=True,
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':5 }
        )
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        plt.show()

    def plot_decision_boundary(self, n, m):
        feature_names = self.data.columns
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, n], X_train[:, m], c=(y_train == 0), cmap=plt.cm.Set1, edgecolor='k') 
        plt.xlabel(feature_names[n])
        plt.ylabel(feature_names[m])
        plt.show()

        logreg = linear_model.LogisticRegression(C=self.bestC)
        logreg.fit(X_train[:, [n, m]], y_train)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, n], X_train[:, m], c=y_train, cmap=plt.cm.Set1, edgecolor='k')
        plt.xlabel(feature_names[n])
        plt.ylabel(feature_names[m])

        X_min, X_max = X_train[:, n].min() - 1, X_train[:, n].max() + 1
        Y_min, Y_max = X_train[:, m].min() - 1, X_train[:, m].max() + 1
        XX, YY = np.meshgrid(np.arange(X_min, X_max, 0.1), np.arange(Y_min, Y_max, 0.1))
        Z = logreg.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX, YY, Z, alpha=0.4, cmap=plt.cm.Set1)
        plt.colorbar()

        plt.scatter(X_train[:, n], X_train[:, m], c=y_train, cmap=plt.cm.Set1, edgecolor='k')
        
        plt.title("Logistic Regression Decision Boundary")
        plt.xlabel(feature_names[n])
        plt.ylabel(feature_names[m])

        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[:, n], X_test[:, m], c=y_test, cmap=plt.cm.Set1, edgecolor='k')
        plt.xlabel(feature_names[n])
        plt.ylabel(feature_names[m])

        X_min, X_max = X_train[:, n].min() - 1, X_train[:, n].max() + 1
        Y_min, Y_max = X_train[:, m].min() - 1, X_train[:, m].max() + 1
        XX, YY = np.meshgrid(np.arange(X_min, X_max, 0.1), np.arange(Y_min, Y_max, 0.1))
        Z = logreg.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX, YY, Z, alpha=0.4, cmap=plt.cm.Set1)
        plt.colorbar()

        plt.scatter(X_train[:, n], X_train[:, m], c=y_train, cmap=plt.cm.Set1, edgecolor='k')
    
        plt.title("Logistic Regression Decision Boundary")
        plt.xlabel(feature_names[n])
        plt.ylabel(feature_names[m])

        plt.show()

        y_pred = logreg.predict(X_test[:, [n, m]])
        print("The logistic regression accuracy score is: ", accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    thing = LogisticRegression()
    thing.setupData()
    # thing.getBestC()
    thing.TBT()
    # thing.correlation_heatmap()
    thing.plot_decision_boundary(1, 20)