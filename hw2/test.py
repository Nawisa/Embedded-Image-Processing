# import required libraries
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('test2.png')
img2 = cv2.imread('test3.png')


# Calculate the histograms, and normalize them
hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
print("Metric Value:", metric_val)

# plot the histograms of two images
plt.subplot(121), plt.hist(img1.ravel(),256,[0,256]),
plt.title('2')
plt.subplot(122), plt.hist(img2.ravel(),256,[0,256]),
plt.title('3')
plt.show()
# Load the images

# Calculate the histograms, and normalize them
hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Find the metric value
metric_val1 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
metric_val2 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CHISQR)
metric_val3 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_INTERSECT)
metric_val4 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
print("Metric Value using Correlation Hist Comp Method", metric_val1)
print("Metric Value using Chi Square Hist Comp Method", metric_val2)
print("Metric Value using Intersection Hist Comp Method", metric_val3)
print("Metric Value using Bhattacharyya Hist Comp Method", metric_val4)