from Models import ConstrainedLR
import pandas
import numpy as np

def run(args):
	df = pandas.read_csv("Test/constrainedregression.csv")
	X = df[["A", "B", "C", "D"]].iloc[0:10].as_matrix()
	y = df["Total"].iloc[0:10].as_matrix()
	model = ConstrainedLR(1)
	model.fit(X, y)

	X =  df[["A", "B", "C", "D"]].iloc[10:].as_matrix()
	y = df["Total"].iloc[10:].as_matrix()
	y_ = model.predict(X)
	mse = np.mean(np.power(y - y_, 2))
	print(model.coef_)
	print(mse)
