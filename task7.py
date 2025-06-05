import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data[:, :2]  
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svc_linear = SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
svc_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
svc_rbf.fit(X_train, y_train)

print("Linear Kernel Classification Report:\n", classification_report(y_test, svc_linear.predict(X_test)))
print("RBF Kernel Classification Report:\n", classification_report(y_test, svc_rbf.predict(X_test)))

print("Linear Kernel Confusion Matrix:\n", confusion_matrix(y_test, svc_linear.predict(X_test)))
print("RBF Kernel Confusion Matrix:\n", confusion_matrix(y_test, svc_rbf.predict(X_test)))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters (RBF):", grid.best_params_)

cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)

def plot_decision_boundary(model, X, y, title):
    h = 0.02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title(title)
    plt.show()

plot_decision_boundary(svc_linear, X_train, y_train, "SVM Linear Kernel Decision Boundary")
plot_decision_boundary(svc_rbf, X_train, y_train, "SVM RBF Kernel Decision Boundary")
