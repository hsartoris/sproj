"""
Simultaneously runs dumb & conv networks and logs both.

Usage: 
    runComp <dataDir> [--r=<runs>] [--s=<samples>] [--lr=<lr>]
        [--outDir=<outDir>] [--runId=<runId>]
    runComp -h | --help

Options:
    -h --help           Show this screen
    <dataDir>           Where to load inputs from. Assumes format dataStaging/!
    --r=<runs>          How many runs. [default: 1]
    --s=<samples>       How many samples to load from <dataDir> [default: 25000].
    --lr=<lr>           Init learning rate [default: .001].
    --outDir=<outDir>   Checkpoint & log directory.
    --runId=<runId>     Specify runId. Defaults to timestamp.
"""
from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
