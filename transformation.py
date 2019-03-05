import cv2 as cv
import numpy as np

img = np.zeros((512,512,3), np.uint8)
img1 = np.zeros((50,126,3), np.uint8)
# img = cv.line(img,(0,0),(511,511),(255,0,0),5)
img = cv.rectangle(img,(200,200),(326,328),(0,255,0),3)

# draw a whole cube's lines
img = cv.line(img,(200,200),(265,150),(0,255,0),3)
img = cv.line(img,(265,150),(391,150),(0,255,0),3)
img = cv.line(img,(391,150),(326,200),(0,255,0),3)
img = cv.line(img,(391,150),(391,278),(0,255,0),3)
img = cv.line(img,(391,278),(326,328),(0,255,0),3)

# draw text on another image first
font = cv.FONT_HERSHEY_SIMPLEX
imgWithText = cv.putText(img,'David', (225, 270), font, 0.8, (0,0,255), 2, cv.LINE_AA)

# draw base text for top and right side usage

cv.putText(img1,'David', (25, 30), font, 0.8, (0,0,255), 2, cv.LINE_AA)
cv.imwrite("text.png", img1)

# get text for top aspect

text = cv.imread("text.png")

pts1 = np.float32([[200,200],[326,328],[326,200]])
pts2 = np.float32([[265,150],[326,200],[391,150]])

M = cv.getAffineTransform(pts1,pts2)
new_text = cv.warpAffine(text, M, (512, 512))

# capture the main text in affined image
new_text = new_text[70:90, 180:250]
imgWithText[160:180, 265:335] = new_text


# get text for right aspect

text = cv.imread("text.png")

pts1 = np.float32([[200,200],[200,328],[326,200]])
pts2 = np.float32([[326,200],[326,328],[391,150]])

M = cv.getAffineTransform(pts1,pts2)
new_text = cv.warpAffine(text, M, (512, 512))

# capture the main text in affined image
new_text = new_text[50:110, 230:270]

imgWithText[220:280, 336:376] = new_text


cv.imwrite("result.png", imgWithText)


