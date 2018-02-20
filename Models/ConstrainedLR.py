from sklearn.linear_model import LinearRegression
import numpy as np

class ConstrainedLR:
	def __init__(self, sum_weight = 1):
		self.__sum_weight = sum_weight
		self.__sklrmodel = LinearRegression(fit_intercept = False)
		self.__trained = False
		self.__n_var = None


	@property 
	def coef_(self):
		if not self.__trained:
			return None

		coef = self.__sklrmodel.coef_
		return np.append(coef, self.__sum_weight - np.sum(coef))
		
	# y = k1X1 + k2X2 + ... + knXn, with sum k1+k2+...+kn = c
	# <=> y = k1X1 + k2X2 + ... + (c - k1 - k2 - ... - kn-1)Xn
	# <=> y - cXn = k1(X1 - Xn) + k2(X2 - Xn) + ... + kn-1(Xn-1 - Xn)
	def __preprocess(self, X, y = None):
		n_var = self.__n_var
		if n_var != X.shape[1]:
			raise Exception("Input dimension (%d) does not match training data (%d)"%(X.shape[1], n_var))
		
		if y is not None:
			y = y - self.__sum_weight*X[:, n_var - 1]
		offset = np.copy(X[:, n_var - 1])
		X[:, 0:(n_var - 1)] = X[:, 0:(n_var - 1)] - X[:, n_var - 1].reshape(-1, 1)
		X[:, n_var - 1] = 1

		return X, y, offset

	def fit(self, X, y):
		if self.__trained:
			raise Exception("model already trained")

		self.__n_var = X.shape[1]
		X, y, _ = self.__preprocess(X, y)
		
		self.__sklrmodel.fit(X[:, 0:(self.__n_var - 1)], y)
		self.__trained = True

	def predict(self, X):
		if not self.__trained:
			raise Exception("model not yet trained")

		X, _, offset = self.__preprocess(X)
		y = self.__sklrmodel.predict(X[:, 0:(self.__n_var - 1)])
		y = y + offset
		return y


		