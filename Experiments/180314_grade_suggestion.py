import pandas
import numpy as np
from Preprocessing.Dataset import Dataset
from Preprocessing.GradeTools import grade2dec
from Tools.IO import make_sure_path_exists
from functools import partial
from sklearn.linear_model import LinearRegression

PREDICTOR_TRAINING_SET = ["1516T1", "1516T2", "1617T1"]
PREDICTOR_TESTING_SET = ["1617T2"]

OUTPUT_DIR = "./output"
'''
We aim to develop a tool to suggest a fair adjustment on teachers' grading by considering
a. student's tendency (cGPA before)
b. student's performance (student's response on survey)
c. teacher's tendency (difference between the mean of grade given by the teacher in this semester and previous semesters)
d. teacher's performance (difference between the mean survey result of the class in this semester and the previous semesters)
d. teacher's relative tendency (difference between the mean grade given by the teacher and other teachers in current semester)
e. teacher's relative performance (difference between the mean survey result of the teacher and other teachers in current semester )
'''

teacher_id_col = "Teacher Name"
cols_performance = ["Q%d (Change)"%i for i in range(1, 18)]
cols_indvar = ["st", "sp", "tt", "tp", "trt", "trp"]
cols_hist = ["tt", "tp"]
col_target = "Grade"

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

def generate_historical_cols(full_df, past_years):
	def historical_cols_helper(df):
		result = {}
		result[teacher_id_col] = df[teacher_id_col].iloc[0]
		result["tt"] = df["Grade"].mean()
		result["tp"] =  df[cols_performance].mean().mean()
		result = pandas.DataFrame({key: [result[key]] for key in result})
		return result

	df = full_df.loc[full_df["Enrollment Term"].isin(past_years)]
	return df.groupby(by = [teacher_id_col], group_keys = False).apply(historical_cols_helper)

def generate_relative_vals(full_df, cur_year):
	result = {}
	df = full_df.loc[full_df["Enrollment Term"] == cur_year]
	result["trt"] = df["Grade"].mean()
	result["trp"] = df[cols_performance].mean().mean()
	return result

def preprocess(df, historical_df, relative_dict):
	result = {}
	cur_term = df["Enrollment Term"].iloc[0]
	grade_mean = df["Grade"].mean()
	performance_mean = df[cols_performance].mean().mean()
	hist_mask = (historical_df[teacher_id_col] == df[teacher_id_col].iloc[0])
	if not hist_mask.any():
		return
	result["Enrollment Term"] = cur_term
	result["Year"] = df["Year of Study"]
	result[teacher_id_col] = df[teacher_id_col].iloc[0]
	result["st"] = df["cGPA (Before)"]
	result["sp"] = df[cols_performance].mean(axis = 1)
	result["tt"] =  grade_mean - historical_df.loc[hist_mask, "tt"].iloc[0]
	result["tp"] =  performance_mean - historical_df.loc[hist_mask, "tp"].iloc[0]
	result["trt"] = grade_mean - relative_dict[cur_term]["trt"]
	result["trp"] = performance_mean - relative_dict[cur_term]["trp"]
	result["Grade"] = df["Grade"]

	return pandas.DataFrame(result)
	

def run(args):
	full_df = pandas.concat([d.df for d in datasets], axis = 0)
	full_df = full_df.reset_index()
	dataset_mask = full_df["Enrollment Term"].isin(PREDICTOR_TRAINING_SET + PREDICTOR_TESTING_SET)
	full_df = full_df.loc[dataset_mask]
	full_df["Grade"] = grade2dec(full_df["Grade"])
	
	historical_df = generate_historical_cols(full_df, PREDICTOR_TRAINING_SET)
	#historical_df.to_csv("historical.csv")
	relative_dict = {term: generate_relative_vals(full_df, term) for term in PREDICTOR_TRAINING_SET + PREDICTOR_TESTING_SET}
	summarized_df = full_df.groupby(by = ["Enrollment Term", teacher_id_col], group_keys = False).apply(partial(preprocess, historical_df = historical_df, relative_dict = relative_dict))
	summarized_df["Predicted"] = np.NAN
	#summarized_df.to_csv("test.csv")

	yr1_students_mask = ~(summarized_df["Enrollment Term"].str.endswith("T1") & (summarized_df["Year"] == 1))
	complete_data_mask = ~(summarized_df[cols_indvar + [col_target]].isnull().any(axis = 1))
	summarized_df = summarized_df.loc[yr1_students_mask & complete_data_mask]

	model = LinearRegression(fit_intercept = False)
	mask_training_data = summarized_df["Enrollment Term"].isin(PREDICTOR_TRAINING_SET)
	mask_testing_data = summarized_df["Enrollment Term"].isin(PREDICTOR_TESTING_SET)
	model.fit(summarized_df.loc[mask_training_data, cols_indvar].as_matrix(), summarized_df.loc[mask_training_data, col_target].as_matrix())
	summarized_df.loc[mask_testing_data, "Predicted"] = model.predict(summarized_df.loc[mask_testing_data, cols_indvar].as_matrix())
	print("r^2 (train) = %.4f"%model.score(summarized_df.loc[mask_training_data, cols_indvar].as_matrix(), summarized_df.loc[mask_training_data, col_target].as_matrix()))
	print("r^2 (test) = %.4f"%model.score(summarized_df.loc[mask_testing_data, cols_indvar].as_matrix(), summarized_df.loc[mask_testing_data, col_target].as_matrix()))
	print(model.coef_)

	make_sure_path_exists(OUTPUT_DIR)
	summarized_df.to_csv(OUTPUT_DIR.rstrip("/") + "/" + "grade_suggestion.csv")