import pandas
from Preprocessing.Dataset import Dataset
from Preprocessing.GradeTools import grade2dec
from Preprocessing.Cleaning import fill_missing
from sklearn.svm import SVR
import Utils
import matplotlib.pyplot as plt
import numpy as np

'''
We aim to predict the grade distribution of a class (measured by mean & sd) by
a. techers' performance (measured by survey: Q1-Q17) 
b. students' past performance (cgpa before, measured by mean & sd)
'''
teacher_id_col = "Teacher Name"
cols_ids = ["Enrollment Term", teacher_id_col]
cols_teachers = ["Q%d (Change)"%i for i in range(1, 18)]
cols_students = ["cGPA (Before)", "Grade"]


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

def calculate_grouped_data(df):
	data = {}
	mask = ~(df["Grade"].isnull())
	df["Grade"].loc[mask] = grade2dec(df["Grade"].loc[mask])

	for key in cols_ids:
		data[key] = df[key].iloc[0]

	for key in cols_teachers:
		data[key] = df[key].mean()

	for key in cols_students:
		data["%s-mean"%key] = df[key].mean()
		data["%s-var"%key] = df[key].var()

	return data

def run():
	full_df = pandas.concat([d.df for d in datasets], axis = 0)

	# Group records by term, teacher and calculate average performance of teachers and students
	groupped_data = full_df.groupby(by = cols_ids, group_keys = False).apply(calculate_grouped_data)
	processed = {key: [groupped_data[i][key] for i in range(len(groupped_data))] for key in groupped_data[0]}
	processed = pandas.DataFrame(processed)

	cols_to_fill = [c for c in processed.columns if c not in cols_ids]
	processed[cols_to_fill] = fill_missing(processed[cols_to_fill], "mean")
	processed.to_csv("Processed.csv")

	grade_mean_model, grade_var_model = SVR(), SVR()
	variable_cols = cols_teachers + ["cGPA (Before)-mean", "cGPA (Before)-var"]
	grade_mean_model.fit(processed[variable_cols], processed["Grade-mean"])
	grade_var_model.fit(processed[variable_cols], processed["Grade-var"])

	# Select data for a specific teacher
	while True:
		teacher_id = Utils.get_input("Teacher id", str, processed[teacher_id_col].unique())
		semester = Utils.get_input("Which term", str, processed["Enrollment Term"].unique())

		pdcols_teacher = [c for c in processed.columns if c in cols_teachers]
		teacher_df = processed.loc[(processed[teacher_id_col] == teacher_id) & (processed["Enrollment Term"] == semester)]
		if teacher_df.shape[0] == 0:
			print("Cannot find the data of the teacher in the specified term")
		else:
			teacher_vector = np.reshape(teacher_df[cols_teachers].iloc[0], len(pdcols_teacher))
			break

	# Plot heat map
	cgpa_before_mean_scale = np.linspace(0, 4, 400)
	cgpa_before_var_scale = np.linspace(0, 1, 100)
	cgpa_before_mean_scale_2d, cgpa_before_var_scale_2d = np.meshgrid(cgpa_before_mean_scale, cgpa_before_var_scale)

	input_matrix = np.zeros((400*100, len(pdcols_teacher) + 2))
	input_matrix[:, 0:len(pdcols_teacher)] = teacher_vector
	input_matrix[:, len(pdcols_teacher)] = np.reshape(cgpa_before_mean_scale_2d, (400*100))
	input_matrix[:, len(pdcols_teacher) + 1] = np.reshape(cgpa_before_var_scale_2d, (400*100))
	
	prediction = grade_mean_model.predict(input_matrix)
	prediction = np.reshape(prediction, (100, 400))
	ax = plt.subplot(1, 1, 1)
	ax.set_title("Mean Grade Heatmap (%s-%s)"%(semester, teacher_id))
	ax.set_xlabel("cGPA (Before) (mean)")
	ax.set_ylabel("cGPA (Before) (var)")
	plt.pcolor(cgpa_before_mean_scale_2d, cgpa_before_var_scale_2d, prediction)
	plt.colorbar()
	plt.show()