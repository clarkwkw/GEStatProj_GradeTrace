import argparse
import importlib

def parse_args():
	parser = argparse.ArgumentParser(description="Loads a test script under Test/")
	parser.add_argument("script", type = str, help = "The name of the script")
	args, extra_args = parser.parse_known_args()
	return args, extra_args

try:
	args, extra_args = parse_args()
	module = importlib.import_module('Test.'+args.script)
	module.run(extra_args)
except ImportError:
	print("> Cannot load script '%s', abort."%args.script)
	exit(-1)
