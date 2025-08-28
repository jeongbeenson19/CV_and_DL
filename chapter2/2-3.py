import cv2 as cv
import sys

img = cv.imread('silver.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_small = cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)

cv.imwrite('silver_gray.png', gray)
cv.imwrite('sliver_gray_small.png', gray_small)

cv.imshow('Color Image', img)
cv.imshow('Gray Image', gray)
cv.imshow('Gray Image Small', gray_small)

cv.waitKey()
cv.destroyAllWindows()
