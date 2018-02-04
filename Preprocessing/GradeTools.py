import bisect

default_grade_table = {
	"A": 4.0,
	"A-": 3.7,
	"B+": 3.3,
	"B": 3.0,
	"B-": 2.7,
	"C+": 2.3,
	"C": 2.0,
	"C-": 1.7,
	"D+": 1.3,
	"D": 1.0,
	"F": 0.0
}

default_dec_grade = sorted(list(default_grade_table.items()), key = lambda x: x[1])
default_dec = [d for _, d in default_dec_grade]

def grade2dec(series):
	return list(map(lambda x: default_grade_table[x], series))

def dec2grade(series):
	return list(map(lambda x: default_dec_grade[bisect.bisect(default_dec, x) - 1][0], series))
	