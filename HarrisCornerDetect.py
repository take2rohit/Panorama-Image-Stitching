import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



class HarrisDetector():

    def __init__(self, path):
        
        self.im = cv2.imread(path,0)


    def sobel(self,custom_im=None, disp=False):
        
        if custom_im is None:
            custom_im = self.im

        sobelGx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelGy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        x_grad = signal.correlate2d(custom_im.copy(),sobelGx,mode='valid')
        y_grad = signal.correlate2d(custom_im.copy(),sobelGy,mode='valid')    
        # f = abs(x_grad) + abs(y_grad)
        
        if disp:
            cv2.namedWindow('x_grad', cv2.WINDOW_NORMAL)
            cv2.namedWindow('y_grad', cv2.WINDOW_NORMAL)
            cv2.imshow('x_grad', x_grad.astype(np.uint8))
            cv2.imshow('y_grad', y_grad.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return x_grad, y_grad


    def findCorners(self,disp=False):

        blur = cv2.GaussianBlur(self.im,(9,9),0)
        x_grad, y_grad = self.sobel(blur)
        h,w = blur.shape[0], blur.shape[1]
        print(h,w)

        Ixx = np.multiply(x_grad,x_grad) 
        Iyy = np.multiply(y_grad,y_grad)
        Iyx = np.multiply(y_grad,x_grad)

        

#read images

Harris = HarrisDetector('img/chess.jpg')
Harris.sobel(disp=False) 
Harris.findCorners(disp=True)
