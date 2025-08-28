import cv2 as cv
import sys

img = cv.imread('silver.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('Image Display', img)

print(type(img))
print(img.shape)
print(img[0, 0, 0], img[0, 0, 1], img[0, 0, 2])

cv.waitKey()
cv.destroyAllWindows()
