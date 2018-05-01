import pandas
import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import norm
import random
from Preprocessing.Dataset import Dataset
from Preprocessing.GradeTools import grade2dec
from Preprocessing.Cleaning import range_to_num
from Models import KNNRegressor
from Tools.IO import make_sure_path_exists

TRAINING_SET = ["1516T1", "1516T2", "1617T1"]
TESTING_SET = ["1617T2"]

OUTPUT_DIR = "./output"

'''
This script is to find out similar students in the past semesters, using:
1. entry survey (Q1 - 17)+ cGPA
2. exit survey  (Q1 - 17 + effort related question) + cGPA
'''

effort_cols = ["Q18 (Assigned Text Read)", "Q19 (Chinese Translation)", "Q22 (% Lecture)"]
cols_entry = ["Q%d (Before)"%i for i in range(1, 18)] + ["cGPA (Before)"]
cols_exit = ["Q%d (After)"%i for i in range(1, 18)] + effort_cols + ["cGPA (Before)"]
extra_info = ["Grade"]

pca_ndim_entry = 5
pca_ndim_exit = 7
top_k = 4

datasets = [
	Dataset(
		"raw_data/201516 Master Data.csv", 
		[
			("raw_data/201516T1 marks.csv", "UGFN"), 
			("raw_data/201516T2 marks.csv", "UGFN")
		],
		keep_common_only = False
	),
	Dataset(
		"raw_data/201617 Master Data.csv", 
		[
			("raw_data/201617T1 marks.csv", "UGFN"), 
			("raw_data/201617T2 marks.csv", "UGFN")
		],
		keep_common_only = False
	)
]

def df_to_matrix(df, cols):
	matrix = np.zeros((df.shape[0], len(cols)))
	c = 0

	for col in cols:
		if col in effort_cols:
			matrix[:, c] = range_to_num(df[col])
		else:
			matrix[:, c] = pandas.to_numeric(df[col], errors = 'coerce')
		c += 1

	col_mean = np.nanmean(matrix, axis = 0)
	inds = np.where(np.isnan(matrix))
	matrix[inds] = np.take(col_mean, inds[1])

	return matrix

def get_grades(df):
	grades = np.asarray(grade2dec(df["Grade"]))
	mean = np.nanmean(grades)
	inds = np.where(np.isnan(grades))
	grades[inds[0]] = mean
	return grades

def run(args):
	full_df = pandas.concat([d.df for d in datasets], axis = 0)
	full_df = full_df.reset_index()

	# Prepare data for entry matrices
	training_mask = full_df["Enrollment Term"].isin(TRAINING_SET)
	training_df = full_df.loc[training_mask]
	entry_training_matrix = df_to_matrix(training_df, cols_entry)
	exit_training_matrix = df_to_matrix(training_df, cols_exit)

	# Prepare data for exit matrices
	testing_mask = full_df["Enrollment Term"].isin(TESTING_SET)
	testing_df = full_df.loc[testing_mask]
	entry_testing_matrix = df_to_matrix(testing_df, cols_entry)
	exit_testing_matrix = df_to_matrix(testing_df, cols_exit)

	# Draw one testing sample to look at nearest neighbors
	test_index = random.choice(range(testing_df.shape[0]))

	# Find similar students on entry data
	regressor = KNNRegressor(n_neighbors = top_k, pca = pca_ndim_entry)
	regressor.fit(entry_training_matrix, get_grades(training_df))
	sim, ind = regressor.kneighbors(entry_testing_matrix[test_index].reshape([1, -1]), n_neighbors = top_k)
	entry_df = pandas.concat([testing_df.iloc[[test_index]], training_df.iloc[ind[0]]])
	entry_df["Similarity"] = np.insert(sim, 0, 1)
	entry_df[cols_entry + extra_info + ["Similarity"]].to_csv("Entry_example.csv")

	testing_df["Entry Prediction"] = regressor.predict(entry_testing_matrix)

	print("Entry R^2 (test) = %.3f"%regressor.score(entry_testing_matrix, get_grades(testing_df)))

	# Find similar students on exit data
	regressor = KNNRegressor(n_neighbors = top_k, pca = pca_ndim_exit)
	regressor.fit(exit_training_matrix, get_grades(training_df))
	sim, ind = regressor.kneighbors(exit_testing_matrix[test_index].reshape([1, -1]), n_neighbors = top_k)
	exit_df = pandas.concat([testing_df.iloc[[test_index]], training_df.iloc[ind[0]]])
	exit_df["Similarity"] = np.insert(sim, 0, 1)
	exit_df[cols_exit + extra_info + ["Similarity"]].to_csv("Exit_example.csv")

	testing_df["Exit Prediction"] = regressor.predict(exit_testing_matrix)

	testing_df[["Grade", "Entry Prediction", "Exit Prediction"]].to_csv("Prediction.csv")

	print("Exit R^2 (test) = %.3f"%regressor.score(exit_testing_matrix, get_grades(testing_df)))

