from .Visualize import matVis, matVis2
import socket
if not socket.gethostname() == "marvin":
    from .MatrixGen import loadMats, initMats, saveMats
