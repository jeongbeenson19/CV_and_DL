import skimage
import numpy as np
import cv2 as cv

img = skimage.data.coffee()
cv.imshow('Coffee image', cv.cvtColor(img, cv.COLOR_RGB2BGR))

slic1 = skimage.segmentation.slic(img, compactness=20, n_segments=600)
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)
sp_img1 = np.uint8(sp_img1*255.0)

slic2 = skimage.segmentation.slic(img, compactness=40, n_segments=600)
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2*255.0)

cv.imshow('Super pixes (compact 20)', cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR)) # type: ignore
cv.imshow('Super pixes (compact 40)', cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR)) # type: ignore

cv.waitKey()
cv.destroyAllWindows()
