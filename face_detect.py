import cv2 as cv

img = cv.imread('Photos\group 2.jpg')
cv.imshow('Original',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
print(f"Number of faces found = {len(face_rect)}")

for (x,y,w,h) in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow("detected faces",img)
cv.waitKey(0)