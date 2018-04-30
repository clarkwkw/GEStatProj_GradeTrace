import pandas
import numpy as np
from Preprocessing.Dataset import Dataset
from Tools.IO import make_sure_path_exists
from Preprocessing.Cleaning import range_to_num
from sklearn.decomposition import PCA
from numpy.linalg import norm
import random

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

pca_ndim_entry = None
pca_ndim_exit = None
top_k = 10

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

def df_to_matrix(df, cols, pca = None):
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

	if type(pca) is PCA:
		matrix = pca.transform(matrix)
	elif type(pca) is int:
		pca = PCA(n_components = pca)
		matrix = pca.fit_transform(matrix)

	return matrix, pca

def similar_data(df, matrix, vect, top_k):
	matrix_norm = norm(matrix, axis = 1)
	vect_norm = norm(vect)
	similarity = np.divide(np.dot(matrix, vect), matrix_norm*vect_norm)
	most_similar_indices = np.argsort(-similarity)[:top_k]
	return df.iloc[most_similar_indices], similarity[most_similar_indices]

def run(args):
	full_df = pandas.concat([d.df for d in datasets], axis = 0)
	full_df = full_df.reset_index()

	# Prepare data for entry matrices
	training_mask = full_df["Enrollment Term"].isin(TRAINING_SET)
	training_df = full_df.loc[training_mask]
	entry_training_matrix, entry_training_pca = df_to_matrix(training_df, cols_entry, pca_ndim_entry)
	exit_training_matrix, exit_training_pca = df_to_matrix(training_df, cols_exit, pca_ndim_exit)

	# Prepare data for exit matrices
	testing_mask = full_df["Enrollment Term"].isin(TESTING_SET)
	testing_df = full_df.loc[testing_mask]
	entry_testing_matrix, _ = df_to_matrix(testing_df, cols_entry, entry_training_pca)
	exit_testing_matrix, _ = df_to_matrix(testing_df, cols_exit, exit_training_pca)

	# Find similar students on entry data
	test_index = random.choice(range(testing_df.shape[0]))
	entry_vect = entry_testing_matrix[test_index, :].reshape([-1])
	entry_df, similarities = similar_data(training_df, entry_training_matrix, entry_vect, top_k)
	entry_df = pandas.concat([testing_df.iloc[[test_index]], entry_df])
	entry_df["Similarity"] = np.insert(similarities, 0, 1)
	entry_df[cols_entry + extra_info + ["Similarity"]].to_csv("Entry_similarity.csv")


	# Find similar students on exit data
	exit_vect = exit_testing_matrix[test_index, :].reshape([-1])
	exit_df, similarities = similar_data(training_df, exit_training_matrix, exit_vect, top_k)
	exit_df = pandas.concat([testing_df.iloc[[test_index]], exit_df])
	exit_df["Similarity"] = np.insert(similarities, 0, 1)
	exit_df[cols_exit + extra_info + ["Similarity"]].to_csv("Exit_similarity.csv")







