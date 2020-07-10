# import progressbar
# from time import sleep
# bar = progressbar.ProgressBar(maxval=20, \
#     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# bar.start()
# for i in range(20):
#     bar.update(i+1)
#     sleep(0.1)
# bar.finish()

# source: https://towardsdatascience.com/least-squares-linear-regression-in-python-54b87fc49e77
# py -i draftplot.py (to keep script open)

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)
print(X.shape)
n = X.shape[1]
r = np.linalg.matrix_rank(X)

U, sigma, VT = np.linalg.svd(X, full_matrices=False)
D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
V = VT.T

X_plus = V.dot(D_plus).dot(U.T)
w = X_plus.dot(y)

error = np.linalg.norm(X.dot(w) - y, ord=2) ** 2
print("error: ", error)

np.linalg.lstsq(X, y, rcond=None)

# plt.scatter(X, y, c='b')
# plt.plot(X, w*X, c='g')

lr = LinearRegression()
lr.fit(X, y)
w = lr.coef_[0]
x_slope = X
y_slope = w*X

plt.scatter(X, y)
plt.plot(X, w*X, c='red')
plt.show()




