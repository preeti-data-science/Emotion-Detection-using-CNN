# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:57:37 2019

@author: Preeti
"""


import cv2,time
import numpy as np #importing opencv and time module

a=1
first_frame = None

face_cascade= cv2.CascadeClassifier("D:/Users/Preeti/Anaconda3/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0) #capturing the video from inbuilt webcam
while(True): #till the time we dont press q ie we are able to capture the video
    
    a=a+1# Capturing  frame-by-frame through this loop so that frames will appear as video
    
    check, frame = video.read() #check is a boolen return true if we are able to capture the image/video,frame is a numpy array
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     #converting each frame into a gray scale image
   # gray_img = cv2.GaussianBlur(gray_img,(21,21),0)
    
    if first_frame is None:  #We will be using them in the future while doingCNN
        first_frame = gray_img
        continue
    
  #  delta_frame = cv2.absdiff(first_frame,gray_img)
  #  thresh_delta = cv2.threshold(delta_frame, 30,255,cv2.THRESH_BINARY)[1]
   # thresh_delta = cv2.dilate(thresh_delta,None,iterations = 0)
    
    faces = face_cascade.detectMultiScale(gray_img , 1.3,5) #Cascade classifier to detect the face
    
    for (x,y,w,h) in faces: #Making Boundaries around face, Green colour,Width is 3
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
       cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3)
        
        
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray_img)

  #  cv2.imshow('delta', delta_frame)
  #  cv2.imshow('thresh', thresh_delta)
    
    key = cv2.waitKey(1) #this will keep on generating new frames after every one millisecond
    
    if key == ord('q'): #press q to release the webcame/camera
        print(a) #will give the number of frames
        break
video.release()
cv2.destroyAllWindows()

# When everything done, release the video
