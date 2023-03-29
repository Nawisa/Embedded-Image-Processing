import cv2
import numpy as np
import matplotlib.pyplot as plt
def getLBPimage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3 
    for ih in range(0,image.shape[0] - neighboor):
        for iw in range(0,image.shape[1] - neighboor):
            ## Step 1: 3 by 3 pixel
            img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
            center       = img[1,1]
            img01        = (img >= center)*1.0
            img01_vector = img01.T.flatten()
            img01_vector = np.delete(img01_vector,4)
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0
            imgLBP[ih+1,iw+1] = num
    
    mask = np.zeros(image.shape[:2],np.uint8)
    # mask[350:450,350:450] = 255
    # mask[500:600,500:600] = 255
    mask[200:300,200:300] = 255
    masked_img = cv2.bitwise_and(image,image,mask=mask)

    hist_full = cv2.calcHist([image],[0],None,[256],[0,256])

    hist_mask = cv2.calcHist([image],[0],mask,[256],[0,256])

    plt.subplot(221),plt.imshow(image,'gray')

    plt.subplot(222),plt.imshow(imgLBP,'gray')

    plt.subplot(223),plt.imshow(masked_img,'gray')

    plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
    cv2.imwrite("test3.png", masked_img[500:600,500:600])

    plt.show()
    return imgLBP
     
image = cv2.imread('hw2.jpg' )
cv2.imshow("test",getLBPimage(image) )
# cv2.imshow("origin",image )
cv2.waitKey()