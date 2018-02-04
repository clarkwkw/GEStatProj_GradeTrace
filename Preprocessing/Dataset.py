import pandas

default_kwargs = {
	"skipinitialspace": True,
	"na_values": ["#VALUE!", "BLANK"],
	"dtype": {
		'ID': str,
		"Valid SID": str
	}
}
marks_cols_to_join = ["Perform", "OD", "Quiz", "RJ", "TP", "Total", "Grade"]

'''
master_file: 
	a string or a tuple of a string (filename) and a dict (kwargs), 
	if kwargs is given, it will be passed to pandas.read_csv

marks_files:
	a tuple of a string (filename), string (subject code: "UGFN"/"UGFH") and a dict (kwargs) (optional)
	if kwargs is given, it will be passed to pandas.read_csv

marks_cols_to_join:
	a list of strings, indicating the columns in marks_files to keep after joining
'''

class Dataset:
	def __init__(self, master_file, marks_files, keep_common_only, marks_cols_to_join = marks_cols_to_join):
		kwargs = default_kwargs
		if type(master_file) is tuple:
			master_file, kwargs_customized = master_file
			kwargs = merge_dict(kwargs_customized, kwargs)

		master_df = pandas.read_csv(master_file, **kwargs)
		
		if "Valid SID" in master_df.columns.values:
			master_df.rename(columns = {"Valid SID": "SID"}, inplace = True)

		master_df["ID"] = master_df["Subject"].str.cat(master_df["SID"], sep = "-")

		master_df.set_index("ID", drop = True, inplace = True)


		marks_dfs = []
		for marks_file in marks_files:
			kwargs = default_kwargs
			if len(marks_file) == 2:
				marks_file, subject = marks_file
			else:
				marks_file, subject, kwargs_customized = marks_file
				kwargs = merge_dict(kwargs_customized, kwargs)
			
			marks_df = pandas.read_csv(marks_file, **kwargs)
			
			if "ID" in marks_df.columns.values:
				marks_df.rename(columns = {"ID": "SID"}, inplace = True)

			marks_df["ID"] = subject + "-" + marks_df["SID"]
			marks_df.set_index("ID", drop = True, inplace = True)

			marks_dfs.append(marks_df)

		marks_df_combined = pandas.concat(marks_dfs, axis = 0)
		
		if marks_cols_to_join is not None:
			marks_df_combined = marks_df_combined[marks_cols_to_join]
		
		marks_df_combined.rename(columns = {colname: "Marks-" + colname for colname in marks_df.columns.values}, inplace = True)
		
		common_indices = master_df.index.intersection(marks_df_combined.index)
		join_method = "inner" if keep_common_only else "left"

		self.__df = master_df.merge(marks_df_combined, how = join_method, left_index = True, right_index = True)
	
	@property 
	def df(self):
		return self.__df

if __name__ == "__main__":
	dataset_1 = Dataset(
		"raw_data/201516 Master Data.csv", 
		[
			("raw_data/201516T1 marks.csv", "UGFN"), 
			("raw_data/201516T2 marks.csv", "UGFN")
		],
		False
	)
	dataset_2 = Dataset(
		"raw_data/201617 Master Data.csv", 
		[
			("raw_data/201617T1 marks.csv", "UGFN"), 
			("raw_data/201617T2 marks.csv", "UGFN")
		],
		False
	)
	dataset_1.df.to_csv("combined_1.csv")
	dataset_2.df.to_csv("combined_2.csv")