import cv2 as cv
import numpy as np
from math import sqrt, log10, ceil, floor

"""
reference: 

Single-Image Vignetting Correction by Constrained
Minimization of log-Intensity Entropy, written by Torsten Rohlfing

"""

def getCorrectedPic(img, params):
    a, b, c = params
    height, width = img.shape
    pic = np.zeros((height, width))
    x_, y_ = int(width/2), int(height/2)

    for i in range(height):
        for j in range(width):
            val = ((i-y_)**2+(j-x_)**2)/(x_**2+y_**2)
            r = sqrt(val)
            g = 1+a*(r**2)+b*(r**4)+c*(r**6)
            pic[i][j] = img[i][j]*g
    return pic

def getOptimizedVal(pic):
    height, width = pic.shape
    I = [[0]*width for _ in range(height)]
    n = [0]*300
    for i in range(height):
        for j in range(width):
            I[i][j] = 255*log10(1+pic[i][j])/log10(256)
            k1, k2 = floor(I[i][j]), ceil(I[i][j])
            # print(pic[i][j], k1, k2)
            n[k1] += (1+k1-I[i][j])
            n[k2] += (k2-I[i][j])
    # TODO: GaussianBlur
    total = sum(n)
    H = 0
    for i in range(len(n)):
        if n[i] > 0:
            H += -((n[i]/total)*log10(n[i]/total))
    return H

def judge(params):
    a, b, c = params
    delta = 4*(b**2)-12*a*c
    cond2, cond3, cond4 = False, False, False

    cond1 = c > 0 and delta <= 0
    cond6 = c == 0 and a*b < 0
    cond5 = c == 0 and a*b == 0 and a+b > 0

    if delta > 0:
        delta = sqrt(4*(b**2)-12*a*c)
        q, q_ = (-2*b+delta)/6*c, (-2*b-delta)/6*c
        cond2 = c > 0 and q <= 0
        cond3 = c > 0 and q_ >= 1
        cond4 = c < 0 and q >= 1 and q_ <= 0

    return cond1 or cond6 or cond5 or cond2 or cond3 or cond4

def optimizing(imgPic):
    a, b, c = 0,0,0
    height, width = imgPic.shape
    finalA, finalB, finalC = a, b, c
    delta = 2
    minVal = getOptimizedVal(imgPic)
    while delta >= 1/256:
        print("The start a,b,c is {},{},{}".format(finalA, finalB, finalC))
        tempA, tempB, tempC = finalA, finalB, finalC
        tempMin = minVal
        if judge((tempA-delta,tempB,tempC)):
            print(1)
            pic = getCorrectedPic(imgPic, (tempA-delta,tempB,tempC))
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA-delta, tempB, tempC
        if judge((tempA,tempB-delta,tempC)):
            print(2)
            pic = getCorrectedPic(imgPic, (tempA,tempB-delta,tempC))
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA, tempB-delta, tempC
        if judge((tempA,tempB,tempC-delta)):
            print(3)
            pic = getCorrectedPic(imgPic, (tempA,tempB,tempC-delta))
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA, tempB, tempC-delta
        # try addition
        if judge((tempA+delta,tempB,tempC)):
            print(4)
            pic = getCorrectedPic(imgPic, (tempA+delta,tempB,tempC))
            print(getOptimizedVal(pic), minVal)
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA+delta, tempB, tempC
        if judge((tempA,tempB+delta,tempC)):
            print(5)
            pic = getCorrectedPic(imgPic, (tempA,tempB+delta,tempC))
            print(getOptimizedVal(pic), minVal)
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA, tempB+delta, tempC
        if judge((tempA,tempB,tempC+delta)):
            print(6)
            pic = getCorrectedPic(imgPic, (tempA,tempB,tempC+delta))
            print(getOptimizedVal(pic), minVal)
            if getOptimizedVal(pic) < minVal:
                minVal = getOptimizedVal(pic)
                finalA, finalB, finalC = tempA, tempB, tempC+delta
        print("The end a,b,c is {},{},{}".format(finalA, finalB, finalC))
        print("the delta is {}".format(delta))
        print("The min entropy is {}".format(minVal))

        if abs(tempMin-minVal)<1e-4:
            delta *= 0.5

    finalPic = getCorrectedPic(imgPic, (finalA,finalB,finalC))
    return finalPic

def wipeOut(img=""):
    pic = cv.imread(img)
    grayPic = cv.cvtColor(pic, cv.COLOR_RGB2GRAY)
    cv.imwrite("./sample_initial.png", grayPic)
    height, width = grayPic.shape
    correctedPic = optimizing(grayPic)
    cv.imwrite("./sample_final.png", correctedPic)

if __name__ == '__main__':
    wipeOut("/path/to/img");