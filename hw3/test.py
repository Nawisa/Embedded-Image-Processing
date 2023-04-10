import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io

### Obtain Binary Image

img = cv2.imread("1.jpg")
img = cv2.resize(img,(700,500))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Extract red channels -> it shows the contrast of stone from background
stone=img[:,:,2]
 
# Threshold image to binary using OTSU. Enforce inversion to set objects as 1 and background as 0.
ret1, thresh = cv2.threshold(stone, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Morphological operations to remove noise - opening
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

### Watershed Segmentation

# sure background
sure_bg = cv2.dilate(opening, kernel,iterations=3)

# sure foreground
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret2, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255,0)

# Unknown region
sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
unknown = cv2.subtract(sure_bg,sure_fg)

# markers
cnt = sorted(cv2.findContours(unknown, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+10
markers[unknown==255] = 0

# watershed 
markers = cv2.watershed(img,markers)

# segment stone
img[markers == -1] = [0,255,255]  
img2 = color.label2rgb(markers, bg_label=0)

# show results
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.title('Stone road Segmentation')
plt.figure()
plt.imshow(img2)
plt.axis('off')
plt.title('Colored segmentation')
plt.show()