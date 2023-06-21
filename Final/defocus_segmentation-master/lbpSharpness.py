import numpy as np
import cv2
import math
import argparse
import pytesseract
import matplotlib.pyplot as plt
# Recognize the characters using Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'F:/Users/User/Tesseract-OCR/tesseract.exe'



parser = argparse.ArgumentParser(description='create test images from raw dicom')
parser.add_argument('--input', help='input image where you want to compute sharpness map', required=True)

args = vars(parser.parse_args())



def im2double(im):
	min_val = np.min(im.ravel())
	max_val = np.max(im.ravel())
	out = (im.astype('float') - min_val) / (max_val - min_val)
	return out



def s(x):
	temp = x>0
	return temp.astype(float)


def lbpCode(im_gray, threshold):
	# width, height = im_gray.shape
	interpOff = math.sqrt(2)/2
	# Alpha Matting Initialization (I): s the process of decomposing an image into foreground and background
	#“foreground” as “sharp” and background as “blurred”.
	I = im2double(im_gray)
	pt = cv2.copyMakeBorder(I,1,1,1,1,cv2.BORDER_REPLICATE)
	right = pt[1:-1, 2:]
	left = pt[1:-1, :-2]
	above = pt[:-2, 1:-1]
	below = pt[2:, 1:-1]
	aboveRight = pt[:-2, 2:]
	aboveLeft = pt[:-2, :-2]
	belowRight = pt[2:, 2:]
	belowLeft = pt[2:, :-2]
	interp0 = right
	interp1 = (1-interpOff)*((1-interpOff) * I + interpOff * right) + interpOff *((1-interpOff) * above + interpOff * aboveRight)

	interp2 = above
	interp3 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * above + interpOff * aboveLeft)

	interp4 = left
	interp5 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * below + interpOff * belowLeft)

	interp6 = below
	interp7 = (1-interpOff)*((1-interpOff) * I + interpOff * right ) + interpOff *((1-interpOff) * below + interpOff * belowRight) 
	# LBP code for a pixel, s(x) >= 1
	#s(np - nc); np = intensities of the P neighbouring pixels located on a circle of radius R centred at nc, nc = the intensity of the central pixel(xc,yc)
	s0 = s(interp0 - I-threshold) 
	s1 = s(interp1 - I-threshold)
	s2 = s(interp2 - I-threshold)
	s3 = s(interp3 - I-threshold)
	s4 = s(interp4 - I-threshold)
	s5 = s(interp5 - I-threshold)
	s6 = s(interp6 - I-threshold)
	s7 = s(interp7 - I-threshold)
	LBP81 = s0 * 1 + s1 * 2+s2 * 4   + s3 * 8+ s4 * 16  + s5 * 32  + s6 * 64  + s7 * 128
	# P = neighbouring pixels (8) localed in circle of radius, R centred = 1
	
	LBP81.astype(int)
	# Bins 0–8 are the counts of the uniform patterns; bin 9 is the count of non-uniform patterns

	U = np.abs(s0 - s7) + np.abs(s1 - s0) + np.abs(s2 - s1) + np.abs(s3 - s2) + np.abs(s4 - s3) + np.abs(s5 - s4) + np.abs(s6 - s5) + np.abs(s7 - s6)
	LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	LBP81riu2[U > 2] = 9

	return LBP81riu2




def lbpSharpness(im_gray, s, threshold):

	lbpmap  = lbpCode(im_gray, threshold)
	window_r = (s-1)//2
	h, w = im_gray.shape[:2]
	map =  np.zeros((h, w), dtype=float)
	lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

	lbpmap_sum = (lbpmap_pad==6).astype(float) + (lbpmap_pad==7).astype(float) + (lbpmap_pad==8).astype(float) + (lbpmap_pad==9).astype(float)
	integral = cv2.integral(lbpmap_sum)
	integral = integral.astype(float)

	map = (integral[s-1:-1, s-1:-1]-integral[0:h, s-1:-1]-integral[s-1:-1, 0:w]+integral[0:h, 0:w])/math.pow(s,2)

	return map




if __name__=='__main__':

	img = cv2.imread(args['input'], cv2.IMREAD_COLOR)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	# Apply thresholding to obtain binary image
	_, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# Apply morphological operations to remove noise and fill gaps
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
	# Find contours of license plate characters
	contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	
	# Extract and recognize characters using PyTesseract OCR
	recognized_text = ''
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if w < 180 and h > 30:
		# if w < 2000 and h > 400:
			roi = img_gray[y:y+h, x:x+w]
			# print(w,h)
			# Calculate LBP sharpness only within the ROI
			sharpness_map = lbpSharpness(roi, 21, 0.016)
			sharpness_map = (sharpness_map - np.min(sharpness_map)) / (np.max(sharpness_map - np.min(sharpness_map))) 
			# normalise to ensures that the sharpness map values are scaled between 0 and 1, and then multiplied by 255 to convert them to the range of 0 to 255 (8-bit unsigned integer).
			sharpness_map = (sharpness_map * 255).astype('uint8')

			# Overlay the sharpness map onto the original image
			# blending operation combines the sharpness map image with the original image
			roi_shape = img[y:y+h, x:x+w].shape[:2]
			sharpness_map_resized = cv2.resize(sharpness_map, (roi.shape[1], roi.shape[0]))
			img[y:y+h, x:x+w] = cv2.addWeighted(img[y:y+h, x:x+w], 0.7, cv2.cvtColor(sharpness_map_resized, cv2.COLOR_GRAY2BGR), 0.3, 0)
			text = pytesseract.image_to_string(roi ,lang='eng', config='--psm 13 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
			recognized_text += text.strip()
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# print('Plate characters: ', recognized_text)
			# print(w,h)

			# Access a pixel value at coordinates (x, y)
			# x = 182
			# y = 91
			# pixel_value = sharpness_map[y, x]

			# # Create a small image with the pixel value
			# pixel_img = np.full((100, 100), pixel_value, dtype=np.uint8)

			# # Display the pixel image
			# cv2.imshow('Pixel', pixel_img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Threshold for determining the sharp region
			# threshold = 200

			# # Create a mask to identify the sharp region
			# sharp_mask = sharpness_map > threshold

			# # Iterate over all pixels and print the coordinates and sharpness value for the sharp region
			# for y in range(sharpness_map.shape[0]):
			# 	for x in range(sharpness_map.shape[1]):
			# 		if sharp_mask[y, x]:
			# 			sharpness_value = sharpness_map[y, x]
			# 			print(f"Sharp pixel at ({x}, {y}), Sharpness value: {sharpness_value}")

			# concat = np.concatenate((img, np.stack((sharpness_map_resized,)*3, -1)), axis=1)

			# Create a mask to segment the sharpness area within the ROI
			# _, sharpness_mask = cv2.threshold(sharpness_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

			# # Apply the mask to the original image
			# roi_bgr = img[y:y+h, x:x+w]
			# sharpness_segmented = cv2.bitwise_and(roi_bgr, roi_bgr, mask=sharpness_mask)

			# # Merge the segmented area with the original ROI
			# segmented_roi = cv2.addWeighted(roi_bgr, 0.7, sharpness_segmented, 0.3, 0)
			# img[y:y+h, x:x+w] = segmented_roi
    	# Ground truth or expected text
		ground_truth = "P688CC"
		# ground_truth = "AZM9590"

		# Calculate the accuracy
		num_correct = sum(1 for a, b in zip(recognized_text, ground_truth) if a == b)
		accuracy = num_correct / len(ground_truth) * 100

	print("Ground Truth: ", ground_truth)
	print("Recognized Text: ", recognized_text)
	print("Accuracy: {:.2f}%".format(accuracy))
	plt.imshow(img)
	# plt.imshow(sharpness_map)
	plt.show()
