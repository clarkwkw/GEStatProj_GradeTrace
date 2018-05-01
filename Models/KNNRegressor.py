from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from numpy.linalg import norm
import numpy as np

# KNN with cosine distance can be transformed to KNN with Euclidean distance
# https://stackoverflow.com/questions/34144632/using-cosine-distance-with-scikit-learn-kneighborsclassifier?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# cosine distance: xT y / (||x|| * ||y||) = (x/||x||)T (y/||y||)
# Euclidean distance: sqrt(xTx + yTy − 2 xTy)
# For normalized x & y, xTx = 1, yTy = 1
# If x and y are in the same direction, dist = 0
# If x and y are opposite, dist = 2

class KNNRegressor:
	def __init__(self, similarity_weight = None, pca = None, **kwargs):
		if "weights" not in kwargs:
			kwargs["weights"] = self.prediction_weight

		self.skregressor = KNeighborsRegressor(**kwargs)

		self.similarity_weight = similarity_weight
		self.pca = pca
		if self.similarity_weight is not None:
			self.similarity_weight = np.asarray(self.similarity_weight)

	def transform_X(self, X, fit = False):
		if len(X.shape) != 2:
			raise Exception("X must be a 2-dimensional matrix")

		if fit:
			if type(self.pca) is int:
				self.pca = PCA(n_components = self.pca)

			if type(self.pca) is PCA:
				X = self.pca.fit_transform(X)
		elif type(self.pca) is PCA:
			X = self.pca.transform(X)


		weight = np.full(X.shape[1], 1) if self.similarity_weight is None else self.similarity_weight
		X = np.multiply(X, weight)
		vect_norm = norm(X, axis = 1)
		X = np.divide(X, vect_norm[:, np.newaxis])

		return X

	def prediction_weight(self, dists):
		return 1-0.5*np.power(dists, 2)

	def fit(self, X, y):
		self.skregressor.fit(self.transform_X(X, fit = True), y)

	def kneighbors(self, X, n_neighbors):
		dists, ind = self.skregressor.kneighbors(self.transform_X(X), n_neighbors, return_distance = True)
		return 1 - 0.5*np.power(dists, 2), ind

	def kneighbors_graph(self, X, n_neighbors):
		return self.skregressor.kneighbors_graph(self.transform_X(X), n_neighbors)

	def predict(self, X):
		return self.skregressor.predict(self.transform_X(X))

	def score(self, X, y):
		return self.skregressor.score(self.transform_X(X), y)

