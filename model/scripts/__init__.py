from .Visualize import matVis, matVis2
try:
    from .MatrixGen import loadMats, initMats, saveMats
except ModuleNotFoundError:
    print("Couldn't load tensorflow")
