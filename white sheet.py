# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:44:01 2020

@author: alina shcherbinina
"""

import cv2
import numpy as np
import math


cam = cv2.VideoCapture(0)

flimit = 250
slimit = 250 

def togray(image):
    return (0.2989*image[:, :, 0]+0.587*image[:, :, 1]+0.114 * image[:, :, 2]).astype('uint8')

 

def fupdate(value):
    global flimit
    flimit = value 
    
def supdate(value):
    global slimit
    slimit = value

def get_dist(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)


def get_sheet_shape(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]

    width1 = get_dist(p3[0], p4[0], p3[1], p4[1])
    width2 = get_dist(p2[0], p1[0], p2[1], p1[1])

    height1 = get_dist(p2[0], p3[0], p2[1], p3[1])
    height2 = get_dist(p1[0], p4[0], p1[1], p4[1])

    max_w = max(int(width1), int(width2))
    max_h = max(int(height1), int(height2))

    return max_w, max_h

def order_points(pts):
    result = np.zeros((4, 2), dtype="f4")

    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top-left
    result[2] = pts[np.argmax(s)]  # bottom-right

    s = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(s)]  # top-right
    result[3] = pts[np.argmax(s)]  # bottom-left

    return result



cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mask", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Paper", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('F', 'Mask', flimit, 255, fupdate)
cv2.createTrackbar('s', 'Mask', slimit, 255, supdate)



while cam.isOpened():
    ret, frame = cam.read()
    
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    mask = cv2.inRange(converted, np.array([80, flimit, 0]),
                       np.array([110, slimit,100]))
 
    mask = cv2.GaussianBlur(mask, (5,5),0)
    
    contrours = cv2.findContours(mask.copy(), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(contrours) > 0:
        
        paper = max(contrours, key=cv2.contourArea)
        
        eps = 0.1 * cv2.arcLength(paper, True)
        approx = cv2.approxPolyDP(paper, eps, True)
        cv2.drawContours(frame, [approx], -1, (23, 130, 146), 3)
        
        for p in approx:
            
            cv2.circle(frame, tuple(*p), 2, (222, 146, 31), 1)
        
        
        if len(approx) == 4:
            
            cv2.drawContours(frame, [approx], -1, (175, 76, 147), 3)
            
            pts1 = approx.reshape(4, 2)
            pts1 = order_points(pts1)
            
            cols, rows = get_sheet_shape(pts1)

            pts2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            
            aff_img = cv2.warpPerspective(frame, M, (cols, rows))
            
            # circle
            
            # hsv_min = np.array((0, 77, 17), np.uint8)
            # hsv_max = np.array((208, 255, 255), np.uint8)
            gray = cv2.cvtColor(aff_img, cv2.COLOR_BGR2GRAY)
            # hsv = cv2.cvtColor( aff_img, cv2.COLOR_BGR2HSV )
            thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU)[1]
            contours0, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours0:
                if len(cnt)>275 and len(cnt)<300:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(aff_img,ellipse,(0,0,255),2)
                    cv2.putText(aff_img, "found", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))

            cv2.imshow("Paper", aff_img)

    
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.imwrite("screen_w.png", np.hstack([frame[:,:,0], mask, aff_img]))
            
                    
cam.release()
cv2.destroyAllWindows()