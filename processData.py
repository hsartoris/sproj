import sys
import subprocess

if len(sys.argv) < 4:
	print("Usage: processData.py <numGraphs> <timesteps> <prefix> [initIdx=0]")
	exit()
numGraphs = int(sys.argv[1]) # will generate numGraphs simplicial AND numGraphs random
timesteps = int(sys.argv[2])
prefix = sys.argv[3]
initIdx = (0 if len(sys.argv) < 5 else int(sys.argv[4]))


#for i in range(initIdx, initIdx + numGraphs):
#	subprocess.call("pipeline/pipe.py " + prefix + "/simplicial/" + str(i) + " " + str(timesteps) + " " +  prefix + 
#"/simplicial/", shell=True)

for i in range(initIdx, initIdx + numGraphs):
	subprocess.call("pipeline/pipe.py " + prefix + str(i) + " " + str(timesteps) + " " + prefix, shell=True)
