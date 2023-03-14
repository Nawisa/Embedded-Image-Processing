import cv2
import numpy as np
from timeit import default_timer as timer



# read image as grayscale
img = cv2.imread('hw1.jpg', cv2.IMREAD_GRAYSCALE)

# threshold to binary
thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]

# apply morphology open with square kernel to remove small white spots
cv2.setUseOptimized(True) #with AVX
start = timer()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 3,5,7,9
morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# apply morphology close with horizontal rectangle kernel to fill horizontal gap
morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel)
end = timer()
print("Kernel:3", end - start) # 3,5,7,9
cv2.imshow("Open", morph1)
cv2.imshow("Close", morph2)
cv2.waitKey(0)

# show results
# cv2.imshow("thresh", thresh)

cv2.setUseOptimized(False) # without AVX
start = timer()
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 3,5,7,9
morph3 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
# apply morphology close with horizontal rectangle kernel to fill horizontal gap
morph4 = cv2.morphologyEx(morph3, cv2.MORPH_CLOSE, kernel1)
end = timer()
print("Kernel:3", end - start) # 3,5,7,9
cv2.imshow("Open1", morph3)
cv2.imshow("Close1", morph4)
cv2.waitKey(0)