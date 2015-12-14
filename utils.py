# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 10:53:27 2015

@author: Bart
"""

from __future__ import division
import numpy as np
import pylab as pl
import scipy
import cv2
import time


RECT_PTS    = np.zeros((2,2))
RECT_CBACK  = None



def computePSR(im):
    _, mval, _, (mx, my) = cv2.minMaxLoc(im)
    smean, sstd = im.mean(), im.std()
    psr = (mval-smean) / (sstd+1e-5)
    return psr


def preprocessing(im):
    res = im.astype(np.float64)
    res = np.log(res+1.0)
    res -= res.mean()
    res /= res.max()
    
    return res


def genGaussianMatrix(width, height, center, sigma):
    gauss_row = np.exp(-(np.arange(0,width) - center[0])**2/(2*sigma**2))
    gauss_col = np.exp(-(np.arange(0,height) - center[1])**2/(2*sigma**2))
    gauss_row = gauss_row.reshape((1, width))
    gauss_col = gauss_col.reshape((height, 1))
    res = gauss_col * gauss_row
    res /= 2*np.pi*sigma**2
    
    return res


def showImage(title, im):
    im = (im - np.min(im))
    im /= np.max(im)
    im *= 255
    im = im.astype(np.uint8)
    cv2.imshow(title, im)


def runCamLoop(cback, camid = 0):
    cap = cv2.VideoCapture(camid)
    starting_time = time.time()
    nb_frames = 0
    while(True):
        ret, frame = cap.read()
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = cback(im)
        
        nb_frames += 1
        current_time = time.time()
        cv2.imshow('frame', im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('f'):
            print "Moyenne de %.02f FPS" % (nb_frames / (current_time - starting_time))
        
        
    cap.release()
    cv2.destroyAllWindows()


def captureRectangle(event, x, y, flags, param):
    global RECT_PTS
    global RECT_CBACK
    
    if event == cv2.EVENT_LBUTTONDOWN:
        RECT_PTS[0,0] = y
        RECT_PTS[0,1] = x
    elif event == cv2.EVENT_LBUTTONUP:
        RECT_PTS[1,0] = y
        RECT_PTS[1,1] = x
        RECT_CBACK(RECT_PTS)
        

def initTemplateCapture(cback):
    global RECT_CBACK
    RECT_CBACK = cback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', captureRectangle)


def drawRectangle(im, pts, cross):
    if not cross:
        c = (255, 255, 255)
    else:
        c = (0,0,0)
    cv2.rectangle(im, (pts[0,1],pts[0,0]),(pts[1,1],pts[1,0]), c, 2)
    if cross:
        cv2.line(im, (pts[0,1],pts[0,0]),(pts[1,1],pts[1,0]), c, 2)
        cv2.line(im, (pts[1,1],pts[0,0]),(pts[0,1],pts[1,0]), c, 2)
        
    
    
    
if __name__ == "__main__":
    im = scipy.misc.lena();
    print im.shape
    pl.imshow(preprocessing(im))
    pl.gray()
    pl.show()
    runCamLoop(preprocessing)