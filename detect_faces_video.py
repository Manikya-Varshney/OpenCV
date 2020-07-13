import cv2
import numpy as np
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
while True:
    _, frame= capture.read()
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rects = face_cascade.detectMultiScale(frame_gray,scaleFactor=1.2, minNeighbors = 5)

    for x,y,w,h in faces_rects:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
 	# if the `q` key was pressed, break from the loop
    if key == ord("q") : break

capture.release()
cv2.destroyAllWindows()
