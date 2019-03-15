import cv2 as cv
import numpy as np

def get_3x3_Directions(i, j):
    return [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]

def LBP(path=""):

    inimage = cv.imread(path)

    grayimg = cv.cvtColor(inimage,cv.COLOR_BGR2GRAY)

    row, col = grayimg.shape
    textureimg = np.zeros((row,col,1), dtype=np.uint8)

    for i in range(1, row-1):
        for j in range(1, col-1):
            temp = np.zeros((8,1,1), dtype=np.int32)
            for index, pos in enumerate(get_3x3_Directions(i, j)):
                x, y = pos
                temp[index] = 1 if grayimg[x][y] > grayimg[i][j] else 0
            textureimg[i][j] = np.packbits(temp)[0]    #将数组中的0-1序列转化成对应的二进制数

    cv.imwrite("texture.jpg", textureimg)
