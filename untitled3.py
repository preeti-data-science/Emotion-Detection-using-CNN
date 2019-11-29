# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:58:18 2019

@author: Preeti
"""

iimport cv2,time
first_frame = None

video = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    check, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
    gray_img = cv2.GaussianBlur(gray_img,(21,21),0)
    if first_frame is None:
        first_frame = gray_img
        continue
    
    delta_frame = cv2.absdiff(first_frame,gray_img)
    thresh_delta = cv2.threshold(delta_frame, 30,255,cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations = 0)
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) <1000:
            continue
        (x,y,w,h)= cv2.boundingRect(contour)
        cv2.Rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray_img)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)
    
    key = cv2.waitKey(1) 
    
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()