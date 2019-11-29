# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:56:46 2019

@author: Preeti
"""

import cv2

img = cv2.imread("d:/Users/Preeti/Desktop/New folder/Preeti_pic.jpeg")

cv2.imshow ("Preeti",img)
           
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("GRAY_SCALE_IMAGE",gray_img)

cv2.waitKey(0)

ret,binary_img = cv2.threshold(img,0,1,cv2.THRESH_BINARY)

cv2.imshow("binary image",binary_img)

cv2.waitKey(0)
    
cv2.destroyAllWindows()


import cv2

import numpy as np

img = cv2.imread("d:/Users/Preeti/Desktop/New folder/Preeti_pic.jpeg")

cv2.imshow ("Preeti",img)
           
cv2.waitKey(0)

B,G,R = cv2.split(img)

zeros = np.zeros(img.shape[:2] , dtype = "uint8")

cv2.imshow("red" , cv2.merge([zeros,zeros,R]))

cv2.waitKey(0)

cv2.imshow("green" , cv2.merge([zeros,G,zeros]))

cv2.waitKey(0)

cv2.imshow("blue" , cv2.merge([B,zeros,zeros]))

cv2.waitKey(0)

cv2.destroyAllWindows()