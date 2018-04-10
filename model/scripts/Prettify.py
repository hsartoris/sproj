import sys
import subprocess

class pretty(object):
	def __init__(self):
		rows, columns = subprocess.check_output(['stty', 'size']).split()
		self.rows = int(rows)
		self.columns = int(columns)

	def write(self, string):
		sys.stdout.write(string)
		sys.stdout.flush()
	
	def arrow(self, curr, maxLen):
		self.write("-"*int(float(curr)*(self.columns-1)/maxLen) + ">\r")

