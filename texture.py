import cv2 as cv
import numpy as np

def get_3x3_Directions(i, j):
    return [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]

def LBP(path=""):

    inimage = cv.imread(path)

    grayimg = cv.cvtColor(inimage,cv.COLOR_BGR2GRAY)

    cv.imwrite("texture.jpg", grayimg)

    row, col = grayimg.shape

    for i in range(row):
        for j in range(col):
            temp = np.zeros((8,1,1), dtype=np.int32)
            for index, pos in enumerate(get_3x3_Directions(i, j)):
                x, y = pos
                if 0<=x<row and 0<=y<col:
                    temp[index] = 1 if grayimg[x][y] > grayimg[i][j] else 0
            grayimg[i][j] = np.packbits(temp)[0]

    cv.imwrite("texture.jpg", grayimg)


if __name__ == '__main__':
    LBP("./girl.jpg")