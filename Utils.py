def get_input(msg, dtype, choices = None):
	value = None
	if choices is not None:
		choices_str = ", ".join(choices)
		print("Choose one from:", choices_str)

	while value is None:
		try:
			typed = input(msg+": ")
			value = dtype(typed)
			if choices is not None and value not in choices:
				raise ValueError
		except ValueError:
			print("Invalid value, try again!")
			
	return value