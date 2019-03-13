import cv2 as cv
import numpy as np

def white_world_filtering(imagePath=""):
    pic = cv.imread(imagePath)
    row, col, channel = pic.shape
    picData = np.reshape(pic, (row*col, channel))
    largest = np.amax(pic, axis=0)
    planes = np.amax(largest, axis=0)
    digal = np.diag([255/planes[0], 255/planes[1], 255/planes[2]])

    for index, pixel in enumerate(picData):
        temp = np.matmul(digal, pixel)
        picData[index] = temp.astype(np.int)
    picData = np.reshape(picData, (row, col, channel))

    cv.imwrite("result.png", picData)

def gray_world_filtering(imagePath=""):
    pic = cv.imread(imagePath)
    row, col, channel = pic.shape
    picData = np.reshape(pic, (row*col, channel))
    total = np.sum(picData, axis=0)
    channelAverage = [val/(row*col) for val in total]
    average = np.sum(channelAverage)/3
    coeffcient = np.array([average/channelAverage[0], average/channelAverage[1], average/channelAverage[2]])
    for index, pixel in enumerate(picData):
        diag = np.diag(pixel)
        picData[index] = np.matmul(diag, coeffcient)
    picData = np.reshape(picData, (row, col, channel))

    cv.imwrite("result.png", picData)
