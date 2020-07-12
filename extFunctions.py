import cv2 as cv


def convertToRGB(image):
	return cv.cvtColor(image, cv.COLOR_BGR2RGB)