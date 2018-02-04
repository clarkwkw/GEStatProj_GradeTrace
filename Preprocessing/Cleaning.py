import pandas

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