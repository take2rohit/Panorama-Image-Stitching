import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



class HarrisDetector():

    def __init__(self ,path, resizer=True):

        self.im = cv2.imread(path)
        if resizer:
            self.im = cv2.resize(self.im, (400,400))
        self.im_bw = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY) 
        

    def sobel(self,custom_im=None, disp=False):
        
        if custom_im is None:
            custom_im = self.im_bw

        sobelGx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelGy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        x_grad = signal.correlate2d(custom_im.copy(),sobelGx,mode='same')
        y_grad = signal.correlate2d(custom_im.copy(),sobelGy,mode='same') 
        x_grad = x_grad / np.max(x_grad)
        y_grad = y_grad / np.max(y_grad)           
        # f = abs(x_grad) + abs(y_grad)
        
        if disp:

            cv2.namedWindow('x_grad', cv2.WINDOW_NORMAL)
            cv2.namedWindow('y_grad', cv2.WINDOW_NORMAL)
            cv2.imshow('x_grad', (x_grad*1).astype(np.uint8))
            cv2.imshow('y_grad', (y_grad*1).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return x_grad, y_grad


    def findCorners(self,k=0.05,thresh_ratio=0.9, disp=False):
        # blur = self.im_bw
        blur = cv2.GaussianBlur(self.im_bw,(3,3),0)
        x_grad, y_grad = self.sobel(blur)
        h,w = blur.shape[0], blur.shape[1]

        Ixx = x_grad * x_grad 
        Iyy = y_grad * y_grad
        Ixy = x_grad * y_grad

        R_image = np.zeros_like(blur,dtype=np.float32)
        
        for i in range(h-10):
            for j in range(w-10):
                Ixx_window_sum = np.sum(Ixx[i:i+10,j:j+10])
                Ixy_window_sum = np.sum(Ixy[i:i+10,j:j+10])
                Iyy_window_sum = np.sum(Iyy[i:i+10,j:j+10])

                M = np.array( [ [Ixx_window_sum, Ixy_window_sum],
                                [Ixy_window_sum, Iyy_window_sum] ] )

                R = np.linalg.det(M) - k * np.trace(M)**2  
                R_image[i+5,j+5] =  R

        max_R = np.max(R_image) * thresh_ratio
        R_image[R_image > max_R] = 255
        R_image[R_image < max_R] = 0

        corner_r, corner_c = np.nonzero(R_image)
        r_prev, c_prev = corner_r[0], corner_c[0]

        for r, c in zip(corner_r, corner_c):
            
            # if ((r - r_prev)**2 + (c - c_prev)**2)**(0.5) >  40:
            cv2.circle(self.im, (c,r), 1, (255,0,255))
    
            r_prev = r
            c_prev = c

        cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('R image thresholded', cv2.WINDOW_NORMAL)
        cv2.imshow('Final Image',self.im)
        cv2.imshow('R image thresholded',R_image.astype(np.uint8))
        cv2.waitKey(0)
        


#read images

Harris = HarrisDetector('img/chess.jpg',resizer=True)
# Harris.sobel(disp=True) 
Harris.findCorners(disp=False)
