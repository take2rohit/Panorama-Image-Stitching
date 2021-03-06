import cv2
import numpy as np
from scipy import signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt



class HarrisDetector():

    def __init__(self, im, resizer=None):

        '''
        Parameters:
            im : image (numpy array)
            resizer : Resize image to (400,400) for fast processing
        '''

        self.im = im
        if resizer is not None:
            self.im = cv2.resize(self.im, resizer)
        self.im_bw = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY) 
        

    def sobel(self, custom_im=None, disp=False):

        '''
        Calculate derivative of the image using sobel operator.

            Parameters
                custom_im : input a different numpy image. 
                            If no image given use class common image
                disp : whether to display results or not
            
            Returns:
                x_grad : numpy array of x gradient image 
                y_grad : numpy array of y gradient image
        '''
        
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


    def findCorners(self, nms_threshold=20, k=0.005, thresh_ratio=0.4, disp=False, kp_size=3, nms1=True):

        '''
        Find corners using Harris detector

            Parameters:
                k : Used in finding R value
                thresh_ratio : High threshold means lesser points  (0 < thresh_ratio < 1)
                               thresh_ratio of 0 means all points whose R value is greater than 0
                               thresh_ratio of 1 means only one point ()
                disp : whether to display results or not
        '''

        blur = cv2.GaussianBlur(self.im_bw,(3,3),0)
        x_grad, y_grad = self.sobel(blur)
        h,w = blur.shape[0], blur.shape[1]

        Ixx = x_grad * x_grad 
        Iyy = y_grad * y_grad
        Ixy = x_grad * y_grad

        sum_kernel = np.ones((9,9)) 
        
        Ixx_window_sum = signal.correlate2d(Ixx.copy(),sum_kernel,mode='valid')
        Ixy_window_sum = signal.correlate2d(Ixy.copy(),sum_kernel,mode='valid')
        Iyy_window_sum = signal.correlate2d(Iyy.copy(),sum_kernel,mode='valid')

        determinant = Ixx_window_sum * Iyy_window_sum - Ixy_window_sum * Ixy_window_sum
        trace = Ixx_window_sum + Iyy_window_sum

        R = determinant - k * trace * trace

        R_image = np.pad(R, (1,4))
        max_R = np.max(R_image) * thresh_ratio

        if nms1:

            R_image[R_image > max_R] = 255
            R_image[R_image < max_R] = 0

            corner_r, corner_c = np.nonzero(R_image)
            corner_r, corner_c = self.nms(corner_r, corner_c, nms_threshold)
        
        else:
            corner_r, corner_c = self.nms2(R_image)

        kp = []
        for r,c in zip(corner_r, corner_c):
            kp.append(cv2.KeyPoint(float(c),float(r),kp_size))

        if disp:

            for r, c in zip(corner_r, corner_c):         
                cv2.circle(self.im, (int(c),int(r)), 5, (255,0,255), thickness=2)

            cv2.namedWindow('R image thresholded', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)

            cv2.imshow('R image thresholded', R_image.astype(np.uint8))
            cv2.imshow('Threshold image', self.im)

            cv2.waitKey(0)
        
        return kp
 
    def nms(self, r, c, distThresh=10):
        
        '''
        Removes points that are closer than DistThresh in L1 space 

            Parameters:
                r, c: Row, Column points
                distThresh: distance threshold to discard points in proximity. 
        '''

        if len(r) == 0:
            return []
        if r.dtype.kind == "i":
            r = r.astype("float")
        if c.dtype.kind == "i":
            c = c.astype("float")

        pick = []

        idxs = np.argsort(c)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            p1_r, p1_c = r[idxs[:last]], c[idxs[:last]]
            p2_r, p2_c = r[i], c[i]
            dist = abs(p2_r - p1_r) + abs(p2_c - p1_c)

            idxs = np.delete(idxs, np.concatenate(([last], np.where(dist < distThresh)[0])))
        return r[pick].astype("int"), c[pick].astype("int")
        

    def nms2(self, data, neighborhoodSize=50, threshold=1):

        '''
        Retains best (highest R_image value) points in the given neighborhood 

            Parameters:
                data: R_image
                neighborhoodSize: size of distance neighborhood. 
                threshold: threshold value to remove small values.
        '''

        data_max = filters.maximum_filter(data, neighborhoodSize)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhoodSize)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)

        return x, y


if __name__=='__main__':

    im = cv2.imread('images/harris_img/chess.jpg')
    Harris = HarrisDetector(im)
    Harris.findCorners(thresh_ratio=0.3, nms_threshold=20, disp=True)
