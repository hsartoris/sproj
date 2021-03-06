def matVis(matrix, outFile = None, width=30, connections=False, n=None, 
        drawText=False, grayScale=False):
    drawMax = False
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    imSize = tuple(dim*width for dim in matrix.shape[::-1])
    yoffset = 0
    if drawText:
        imSize = (imSize[0], imSize[1] + width)
        yoffset = width
    matImg = Image.new('RGBA', imSize, "black")
    white = Image.new('RGBA', matImg.size, 'white')
    pixels = matImg.load()
    pixRange = max(np.max(matrix), abs(np.min(matrix)))

    matArr = np.array(matImg)
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matVal = matrix[i,j]
            pixW = j * width
            pixH = i * width
            matArr[pixH+yoffset:pixH+width+yoffset,pixW:pixW+width] = [
                    (255 if matVal > 0 and not grayScale else 0), 0,
                    (255 if matVal < 0 and not grayScale else 0), 
                    int(abs(matVal)/pixRange * 255)]
            matArr[yoffset:matrix.shape[0]*width, pixW] = [(0, 0, 0, 255)]
            matArr[pixH+yoffset, 0:matrix.shape[1]*width] = [(0,0,0,255)]
            if matVal == pixRange or abs(matVal) == pixRange:
                maxBlock = (i,j)
    # draw vertical border left
    matArr[yoffset:, 0] = [(0,0,0,255)]
    # draw same right
    matArr[yoffset:, len(matArr[0])-1] = [(0,0,0,255)]
    # draw horizontal border top
    matArr[yoffset, :] = [(0,0,0,255)]
    matArr[len(matArr) - 1, :] = [(0,0,0,255)]
    matImg = Image.fromarray(matArr)
    matImg = Image.alpha_composite(white, matImg)
    if drawText:
        fontSize = 1
        font = ImageFont.truetype('/usr/share/fonts/TTF/cmr10.ttf', fontSize)
        testChar = "1"
        if connections and n: testChar = "[0,0]"
        while max(font.getsize(testChar)[0], font.getsize(testChar)[1]) < (width 
            * .6):
            fontSize += 1
            font = ImageFont.truetype('/usr/share/fonts/TTF/cmr10.ttf', 
                    fontSize)
        fontSize -= 1
        font = ImageFont.truetype('/usr/share/fonts/TTF/cmr10.ttf', fontSize)

        draw = ImageDraw.Draw(matImg)

        vertIdx = (matrix.shape[0] - 1) * width + int(width/2)
        for j in range(matrix.shape[1]):
            if connections and n: text = "[" + str(int(j/n)) + "," + str(j%n) + "]"
            else: text = str(j)
            tw, th = draw.textsize(text, font=font)
            #if connections: tw = tw/1.5
            draw.text((j*width + int(width/2) - tw/2, vertIdx - th/2), 
                    text, font=font, fill=(0,0,0,255))

        if maxBlock and drawMax:
            text = str(np.round(matrix[maxBlock], 2))
            tw, th = draw.textsize(text, font=font)
            draw.text((maxBlock[1]*width + tw/2, maxBlock[0]*width + th/2), text, font=font,
                    fill=(0,0,0,255))
    elif not outFile is None:
        f = open(outFile + ".max", "w+")
        f.write(str(np.round(matrix[maxBlock], 2)) + "\n")
        f.close()

    if outFile is None:
        matImg.show()
    else:
        matImg.save(outFile)


def matVis2(matrix, outFile = None, width=15):
    #numpy version
    # not up to date
    from PIL import Image
    import numpy as np
    matImg = Image.new('RGBA', tuple(dim*width for dim in matrix.shape[::-1]),
        "black")
    white = Image.new('RGBA', matImg.size, 'white')
    pixels = matImg.load()
    pixRange = max(np.max(matrix), abs(np.min(matrix)))

    matArr = np.array(matImg)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matVal = matrix[i,j]
            pixW = j * width
            pixH = i * width
            matArr[pixH:pixH+width,pixW:pixW+width] = [(255 if matVal > 0 else 0), 0,
                    (255 if matVal < 0 else 0), 
                    int(abs(matVal)/pixRange * 255)]
            matArr[0:matrix.shape[0]*width, pixW] = [(0, 0, 0, 255)]
    matImg = Image.fromarray(matArr)
    matImg = Image.alpha_composite(white, matImg)
    if outFile is None:
        matImg.show()
    else:
        matImg.save(outFile)
