import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from linearRegression import Linear_regression
# cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# iris = datasets.load_iris()
# X,y = iris.data, iris.target

X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state = 1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], y, color = 'b', marker = 'o', s = 30)
plt.show()
plt.savefig("mygraph.png")


reg = Linear_regression(lr = 0.01)
reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)

def mean_squared_error(Y_test, predictions):
    return np.mean((Y_test - predictions)**2)

mse = mean_squared_error(Y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train,Y_train, color = cmap(0.9), s=10)
m1 = plt.scatter(X_test,Y_test, color = cmap(0.5), s=10)
plt.plot(X, y_pred_line, color = 'black', linewidth = 2, label = 'Prediction')
plt.show()
plt.savefig("finalGraph.png")