import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from extFunctions import *

# importing the image
test_image = cv.imread('baby.jpeg')

# converting the image into grayscale and storing in
# a separate variable

test_image_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)


# importing the haarcascade into a variable
haarcascade_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rects = haarcascade_face.detectMultiScale(test_image_gray,scaleFactor=1.2, minNeighbors = 5)

# Printing the number of faces detected
print('Faces found: ',len(faces_rects))

for x,y,w,h in faces_rects:
	cv.rectangle(test_image, (x,y),(x+w,y+h),(0,255,0),2)

# displaying the image using matplotlib imshow

plt.imshow(convertToRGB(test_image))

# plt.imshow(test_image_gray, cmap = 'gray')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

