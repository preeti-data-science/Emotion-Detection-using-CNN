# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:19:53 2019

@author: Preeti
"""

import cv2
img = cv2.imread('D:/Users/Preeti/Documents/6 Oct AI/mandrill.tif')
cv2.imshow('Input image', img)
cv2.waitKey(0)

import cv2
gray_img = cv2.imread('D:/Users/Preeti/Documents/6 Oct AI/mandrill.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)

cv2.imwrite('output.tif', gray_img)
import os
os.getcwd()

import cv2
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
cv2.imwrite('output1.png', img, [cv2.IMWRITE_PNG_COMPRESSION])

import cv2
print([x for x in dir(cv2) if x.startswith('COLOR_')]) # RGB, CMYK, YUV, HSV - Hue, Saturation,Value # cyan, magenta, yellow, and key, YUV, Y is the luminance (brightness) component while U and V are the chrominance (color) components

import cv2
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey(0)
    
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(yuv_img)
cv2.imshow('Y channel', y)
cv2.imshow('U channel', u)
cv2.imshow('V channel', v)
cv2.waitKey(0)

cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey(0)

img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_COLOR)
g,b,r = cv2.split(img)
gbr_img = cv2.merge((g,b,r))
rbr_img = cv2.merge((r,b,r))
cv2.imshow('Original', img)
cv2.imshow('GRB', gbr_img)
cv2.imshow('RBR', rbr_img)
cv2.waitKey(0)



#Image translation
import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR) #INTER_CUBIC, INTER_NEAREST
cv2.imshow('Translation', img_translation)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70, num_rows + 110))
cv2.imshow('Translation', img_translation)
cv2.waitKey(0)

img_translation = cv2.warpAffine(img_translation, translation_matrix,(num_cols + 70 + 30, num_rows + 110 + 50))
cv2.imshow('Translation', img_translation)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR, cv2.BORDER_WRAP, 1)
cv2.imshow('Translation', img_translation)
cv2.waitKey(0)


import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,int(0.5*num_cols)], [0,1,int(0.5*num_rows)] ])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)
img_translation = cv2.warpAffine(img, translation_matrix, (2*num_cols, 2*num_rows))
img_rotation = cv2.warpAffine(img_translation, rotation_matrix,(num_cols*2, num_rows*2))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey(0)

import cv2
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey(0)

#Affine Transformations Euclidean Transformation
import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0],[int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)

# Projective transformation
import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1],[int(0.66*cols),rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [0,rows-1], [cols/2,0],[cols/2,rows-1]])
dst_points = np.float32([[0,100], [0,rows-101],[cols/2,0],[cols/2,rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)
wait = True 

import cv2
import numpy as np
import math
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        offset_y = 0
        if j+offset_x < rows:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Input', img)
cv2.imshow('Vertical wave', img_output)
cv2.waitKey(0)

import cv2
import numpy as np
import math
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = 0
        offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
        if i+offset_y < rows:
            img_output[i,j] = img[(i+offset_y)%rows,j]
        else:
            img_output[i,j] = 0
cv2.imshow('Horizontal wave', img_output)

import cv2
import numpy as np
import math
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
        if i+offset_y < rows and j+offset_x < cols:
            img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Multidirectional wave', img_output)

import cv2
import numpy as np
import math
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/mandrill.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols)))
        offset_y = 0
        if j+offset_x < cols:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Concave', img_output)
cv2.waitKey(0)

#Blurring is called as low pass filter. Why?
import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
rows, cols = img.shape[:2]
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32) / 9.0 # Divide by 9 to normalize the kernel
kernel_5x5 = np.ones((5,5), np.float32) / 25.0 # Divide by 25 to normalize the kernel
cv2.imshow('Original', img)
output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('Identity filter', output)
output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3 filter', output)
output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x5 filter', output)
cv2.waitKey(0)
output = cv2.blur(img, (3,3))
cv2.imshow('blur', output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
cv2.imshow('Original', img)
size = 15
# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)
cv2.imshow('Motion Blur', output)
cv2.waitKey(0)


import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
cv2.imshow('Original', img)
# generating the kernels
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]]) / 8.0
# applying different kernels to the input image
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
cv2.imshow('Sharpening', output_1)
cv2.imshow('Excessive Sharpening', output_2)
cv2.imshow('Edge Enhancement', output_3)
cv2.waitKey(0)

import cv2
import numpy as np
img_emboss_input = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
# generating the kernels
kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])
# converting the image to grayscale
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY)
# applying the kernels to the grayscale image and adding the offset toproduce the shadow
output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128
output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128
output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128
cv2.imshow('Input', img_emboss_input)
cv2.imshow('Embossing - South West', output_1)
cv2.imshow('Embossing - South East', output_2)
cv2.imshow('Embossing - North West', output_3)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
# It is used to indicate depth of cv2.CV_64F.
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# Kernel size can be: 1,3,5 or 7.
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.waitKey(0)


import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
# It is used depth of cv2.CV_64F.
laplacian = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow('Original', img)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
canny = cv2.Canny(img, 50, 240)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp', 0)
kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
rows, cols = img.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(cols,200)
kernel_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(img)
# applying the mask to each channel in the input image
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask
cv2.imshow('Original', img)
cv2.imshow('Vignette', output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
rows, cols = img.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(int(1.5*cols),200)
kernel_y = cv2.getGaussianKernel(int(1.5*rows),200)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
mask = mask[int(0.5*rows):, int(0.5*cols):]
output = np.copy(img)
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask
cv2.imshow('Input', img)
cv2.imshow('Vignette with shifted focus', output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp', 0)
# equalize the histogram of the input image
histeq = cv2.equalizeHist(img)
cv2.imshow('Input', img)
cv2.imshow('Histogram equalized', histeq)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/plane.bmp')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/Median filter.png')
output = cv2.medianBlur(img, ksize=7)
cv2.imshow('Input', img)
cv2.imshow('Median filter', output)
cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('C:/Users/Innodatatics/Desktop/Machine Learning, AI & DL/Images/G_B filter.jpg')
img_gaussian = cv2.GaussianBlur(img, (13,13), 0) # Gaussian Kernel Size 13x13
img_bilateral = cv2.bilateralFilter(img, 13, 70, 50)
cv2.imshow('Input', img)
cv2.imshow('Gaussian filter', img_gaussian)
cv2.imshow('Bilateral filter', img_bilateral)
cv2.waitKey(0)


