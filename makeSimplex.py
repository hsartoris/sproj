from scripts.GraphKit import Simplex
import sys

if (len(sys.argv) < 3):
	print("Takes args for size of simplex and name")
	exit()
print("Saving " + str(sys.argv[1]) + "-simplex to " + str(sys.argv[2])) 
s = Simplex(int(sys.argv[1]))
s.save(sys.argv[2])
