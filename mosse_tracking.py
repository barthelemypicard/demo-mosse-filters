# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:21:21 2015

@author: Bart
"""

from __future__ import division
import utils
import numpy as np
import pylab as pl
import scipy
import cv2
from scipy import fftpack


TPL_RECT        = None
COS_WINDOW      = None

FILTER_INIT     = False
FILTER_NUM      = 0.0
FILTER_DEN      = 1.0
FILTER_STD      = 2.0
LEARNING_RATE   = 0.125
REG_FACTOR      = 0.01
TPL_CENTER      = np.zeros(2)
PSR_THRES       = 6.0


def initFilter(tpl):
    global COS_WINDOW
    global FILTER_INIT
    global FILTER_STD
    global FILTER_NUM
    global FILTER_DEN
    
    height, width = tpl.shape
    ptpl = tpl * COS_WINDOW
    output = utils.genGaussianMatrix(width, height, (width/2, height/2), FILTER_STD)
    ftpl = np.fft.fft2(ptpl)
    foutput = np.fft.fft2(output)
    FILTER_NUM = foutput * np.conj(ftpl)
    FILTER_DEN = ftpl * np.conj(ftpl)
    FILTER_INIT = True
    
    psr = utils.computePSR(output)
    #W print psr

def tplCback(pts):
    global TPL_RECT
    global COS_WINDOW
    global FILTER_INIT
    
    if TPL_RECT == None:
        TPL_RECT = pts.astype(np.int64)
        width = pts[1, 1] - pts[0, 1]
        height = pts[1, 0] - pts[0, 0]
        cos_col = np.sin(np.pi*np.arange(0, height)/(height-1.0)).reshape((height, 1))
        cos_row = np.sin(np.pi*np.arange(0, width)/(width-1.0)).reshape((1, width))
        COS_WINDOW = cos_col * cos_row
        FILTER_INIT = False
        print pts
    
    
def updateFilter(tpl):
    global COS_WINDOW
    global FILTER_STD
    global FILTER_NUM
    global FILTER_DEN
    global LEARNING_RATE
    
    height, width = tpl.shape
    ptpl = tpl * COS_WINDOW
    output = utils.genGaussianMatrix(width, height, (width/2, height/2), FILTER_STD)
    ftpl = np.fft.fft2(ptpl)
    foutput = np.fft.fft2(output)
    FILTER_NUM = LEARNING_RATE * (foutput * np.conj(ftpl)) + (1.0 - LEARNING_RATE) * FILTER_NUM
    FILTER_DEN = LEARNING_RATE * (ftpl * np.conj(ftpl))+ (1.0 - LEARNING_RATE) * FILTER_DEN
    
    
def processFrame(im):
    utils.preprocessing(im)
    #To see in the right way
    im = cv2.flip(im, 1)
    im_height, im_width = im.shape
    global TPL_RECT
    global TPL_CENTER
    global FILTER_INIT
    global FILTER_NUM
    global FILTER_DEN
    global COS_WINDOW
    global PSR_THRES
    
    psr_test = False
    if TPL_RECT != None:
        tplc = TPL_RECT.astype(np.int64)
        patch = im[tplc[0,0]:tplc[1,0], tplc[0,1]:tplc[1,1]];
        if not FILTER_INIT:
            initFilter(patch)
            print tplc.shape
        else:
            if patch.shape == COS_WINDOW.shape:
                height, width = patch.shape
#                ptpl = patch * COS_WINDOW
#                height, width = ptpl.shape
#                output = utils.genGaussianMatrix(width, height, (width/2, height/2), 2.0)
#                ftpl = np.fft.fft2(ptpl)
#                foutput = np.fft.fft2(output)
#                n = foutput #* np.conj(ftpl)
#                d = ftpl #* np.conj(ftpl)
                G = np.conj(FILTER_NUM/FILTER_DEN) * np.conj(np.fft.fft2(patch * COS_WINDOW))
                g = np.real(np.fft.ifft2(G))
                utils.showImage('output', g);
                utils.showImage('filter', np.real(np.fft.fftshift(np.fft.ifft2(np.conj(FILTER_NUM/FILTER_DEN)))));
                psr = utils.computePSR(g)
                psr_test = psr > PSR_THRES
                if True:
                    peak_pos = np.argmax(g);
                    dy = peak_pos // width - height//2
                    dx = peak_pos % width - width//2
                    if tplc[0,0] - dy < 0:
                        dy = tplc[0,0]
                    if tplc[1,0] - dy >= im_height:
                        dy = tplc[1,0] - im_height
                    if tplc[0,1] - dx < 0:
                        dx = tplc[0,1]
                    if tplc[1,1] - dx >= im_width:
                        dx = tplc[1,1] - im_width
                    tplc[:,0] -= dy
                    tplc[:,1] -= dx
                    TPL_RECT[:,0] -= dy
                    TPL_RECT[:,1] -= dx
                    new_patch = im[tplc[0,0]:tplc[1,0], tplc[0,1]:tplc[1,1]];
                    updateFilter(new_patch)
                # print psr

        utils.drawRectangle(im, tplc, not psr_test)
    
    return im


if __name__ == "__main__":
    utils.initTemplateCapture(tplCback)
    utils.runCamLoop(processFrame)