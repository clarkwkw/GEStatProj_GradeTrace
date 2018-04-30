import pandas
import re
import numpy as np

fill_missing_methods = ["mean"]
def fill_missing(df, fill_method):
	if fill_method not in fill_missing_methods:
		raise Exception("Unexpected method '%s'"%fill_method)
		
	def fill_mean_helper(series):
		mask = series.isnull()
		fill_value = 0
		if fill_method == "mean":
			fill_value = series.mean()

		series.loc[mask] = fill_value
		return series

	return df.apply(fill_mean_helper, axis = 1)

# Convert catagorical response of ranges to numbers
# e.g. if the category contains 2 numbers, it will take the mean
#      if the category contains only 1 number, it will take that number
# for example, '10 to 11' -> 10.5, '12' -> 12
def range_to_num(series):
	range_regex = r"^[^\d]*?(-?\d+(?:\.\d+)?)[^\d]+?(-?\d+(?:\.\d+)?)[^\d]*$"
	range_parser = re.compile(range_regex)

	value_regex = r"^[^\d]*(-?\d+(?:\.\d+)?)[^\d]*$"
	value_parser = re.compile(value_regex)

	result = []
	for value in series:
		if type(value) is not str:
			result.append(np.NAN)
			continue

		obj = range_parser.match(value)
		if obj is not None:
			lower, upper = float(obj.group(1)), float(obj.group(2))
			result.append(0.5*(lower + upper))

		else:
			obj = value_parser.match(value)
			if obj is not None:
				result.append(float(obj.group(1)))
			else:
				result.append(np.NAN)

	return np.asarray(result)