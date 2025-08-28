import cv2 as cv
import sys

img = cv.imread('silver.png')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.rectangle(img, (830, 30), (1000, 20), (0, 0, 255), 2)

cv.putText(img, 'sliver', (830, 24),
           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv.imshow('Draw', img)

cv.waitKey()
cv.destroyAllWindows()
