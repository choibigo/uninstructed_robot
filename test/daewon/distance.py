import cv2
import numpy as np


map_image = cv2.imread('rsint.png')


kernel = np.ones((3, 3), np.uint8)
map_image = cv2.erode(map_image, kernel, iterations=9)
map_image = cv2.dilate(map_image, kernel, iterations=9)

map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)

distance_transform = cv2.distanceTransform(map_image, cv2.DIST_L2, 5)
result = cv2.normalize(distance_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2.imshow('result', result)
cv2.imwrite('result.png', result)
cv2.waitKey()