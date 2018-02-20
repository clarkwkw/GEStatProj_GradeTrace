import pandas
import numpy as np
from Preprocessing.Dataset import Dataset
from Preprocessing.GradeTools import grade2dec
from Preprocessing.Cleaning import fill_missing
from Tools.Visualization import generate_animated_gif
from Tools.IO import make_sure_path_exists
from Models import ConstrainedLR

PREDICTOR_TRAINING_SET = ["1516T1", "1516T2", "1617T1"]
PREDICTOR_TESTING_SET = ["1617T2"]
ADVICE_RATE = 1.0

OUTPUT_DIR = "./output"
'''
We aim to develop a tool to suggest a fair adjustment on teachers' grading by considering
a. student's tendency (cGPA before)
b. student's performance (student's response on survey)
c. teacher's tendency (mean of grade given by the teacher)
d. teacher's performance (the class's response on survey)
'''

teacher_id_col = "Teacher Name"
cols_ids = ["Enrollment Term", teacher_id_col]
cols_performance = ["Q%d (Change)"%i for i in range(1, 18)]
cols_indvar = ["st", "sp", "tt", "tp"]
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

def summarize_input(df):
	data = {}

	if df["Enrollment Term"].iloc[0].endswith("T1"):
		mask = df["Year of Study"] != 1
	else:
		mask = np.full(df.shape[0], True)

	data[col_target] = grade2dec(df.loc[mask, col_target])
	data["st"] = df.loc[mask, "cGPA (Before)"]
	data["sp"] = df.loc[mask, cols_performance].mean(axis = 1)
	data["tt"] = np.nanmean(data[col_target])
	data["tp"] = df.loc[mask, cols_performance].mean().mean()
	for col in cols_ids:
		data[col] = df.loc[mask, col]
	result = pandas.DataFrame(data)

	return result

def run(args):
	full_df = pandas.concat([d.df for d in datasets], axis = 0)
	full_df = full_df.reset_index()
	summarized_data = full_df.groupby(by = cols_ids, group_keys = False).apply(summarize_input)
	summarized_data = summarized_data.dropna(axis = 0, how = "any")
	summarized_data["Adj"] = np.NAN
	
	predictor = ConstrainedLR(sum_weight = 1)
	mask_training_data = summarized_data["Enrollment Term"].isin(PREDICTOR_TRAINING_SET)
	mask_testing_data = summarized_data["Enrollment Term"].isin(PREDICTOR_TESTING_SET)
	predictor.fit(summarized_data.loc[mask_training_data, cols_indvar].as_matrix(), summarized_data.loc[mask_training_data, col_target].as_matrix())

	predicted_grade = predictor.predict(summarized_data.loc[mask_testing_data, cols_indvar].as_matrix())
	adj = -1*ADVICE_RATE*(summarized_data.loc[mask_testing_data, col_target] - predicted_grade)
	summarized_data.loc[mask_testing_data, "Adj"] = adj
	
	print("Weights of", cols_indvar, ":")
	print(predictor.coef_)
	
	make_sure_path_exists(OUTPUT_DIR)
	summarized_data.to_csv(OUTPUT_DIR.rstrip("/") + "/" + "grade_suggestion.csv")