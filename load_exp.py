import argparse
import importlib

def parse_args():
	parser = argparse.ArgumentParser(description="Loads an experiement script under Experiments/")
	parser.add_argument("script", type = str, help = "The name of the script")
	args = parser.parse_args()
	return args

try:
	args = parse_args()
	module = importlib.import_module('Experiments.'+args.script)
	module.run()
except ImportError:
	print("> Cannot load script '%s', abort."%args.script)
	exit(-1)
