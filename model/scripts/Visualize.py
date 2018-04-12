def matViz(matrix, outFile = None, blockWidth=15):
    from PIL import Image
    import numpy as np
    matImg = Image.new('RGBA', tuple(dim*blockWidth for dim in matrix.shape[::-1]),
        "black")
    pixels = matImg.load()
    pixRange = max(np.max(matrix), abs(np.min(matrix)))
    for i in range(matImg.size[0]):
        for j in range(matImg.size[1]):
            # matrix version: [j,i]
            matVal = matrix[int(j/blockWidth), int(i/blockWidth)]
            pixels[i,j] = ((255 if matVal > 0 else 0), 0,
                    (255 if matVal < 0 else 0), int(abs(matVal)/pixRange * 255))
            if i%blockWidth == 0:
                #border
                pixels[i,j] = (255,255,255,255)
    matImg.show()
