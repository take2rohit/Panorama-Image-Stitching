import cv2
import numpy as np
import os, glob

class Panorama():

    def __init__(self,image_folder,save_path=None, show_imgs=False):

        '''
        Parameters:
            image_folder : Folder consisting of many images to be 
                            be stitched horizontally
            save_path : Directory to save panorama image. If directory 
                        doesnt exists, code will create automatically
            show_imgs : Show input images which are to be stitches
        '''

        self.image_folder = image_folder
        img_path = sorted(glob.glob(os.path.join(self.image_folder,'*')))
        self.images = [cv2.imread(p) for p in img_path]
        self.bw_images = [cv2.imread(p,0) for p in img_path]
        self.save_path = save_path

        if show_imgs:
            for c, img in enumerate(self.images):
                cv2.namedWindow(str(c),cv2.WINDOW_NORMAL)
                cv2.imshow(str(c), img)
            print('\nShowing a set of all images')
            print('Press any key to create a Panorama',end='\n \n')
            cv2.waitKey(0)

    def createCanvas(self, padding = 30):

        '''
        Creates a Canvas with first image appended in it. 
        Canvas dimension 
            height => height of first image + padding
            width => sum of width of all image

        '''

        img1_shp = self.images[0].shape
        width = 0
        for im in self.images:
            width += im.shape[1]

        cnv_shp = (img1_shp[0]+padding,width,img1_shp[2])
        canvas = np.zeros(cnv_shp,np.uint8) 
        canvas[0:img1_shp[0],0:img1_shp[1]] = self.images[0]

        return canvas

    def image_stitch(self, img1,img2):

        grayImg1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        grayImg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        flg = np.array((grayImg1,grayImg2))
        idx = np.argmax(flg,axis=0)
        id1 = np.where(idx ==0)
        id2 = np.where(idx==1)
        img = np.zeros_like(img1)
        img[id1] = img1[id1]
        img[id2] = img2[id2]
        return  img

    def findSIFTfeatures(self,img1,img2, top_n=25, show_match=False):

        '''
        Find good features to match using SIFT

        Parameters:
            img1 : First numpy image
            img2 : Second numpy image
            top_n : State how many top n features needed (default - 25)
            show_match : Display image of matching
        
        Returns
            (kp1, kp2) : Keypoints of img1 and img2 respectively
            best_n : list of best n keypoints (default - 25)
        '''
    
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good_features = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_features.append(m)

        
        best_n = sorted(good_features, key=lambda x: x.distance)[:top_n]

        if show_match:
            best_n_plot = [[m] for m in best_n]
            match_pt_img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,best_n_plot,None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('SIFT matched images', match_pt_img)

        return (kp1, kp2), best_n

    def createPanaroma(self, padding = 100):

        canvas = self.createCanvas(padding = padding)
        (kp1, kp2), best_n = self.findSIFTfeatures(self.bw_images[0],self.bw_images[1])

        img1_pts = np.float32([kp1[val.queryIdx].pt for val in best_n]).reshape(-1,1,2)
        img2_pts = np.float32([kp2[val.trainIdx].pt for val in best_n]).reshape(-1,1,2)
        
        H,mask = cv2.findHomography(img2_pts,img1_pts,cv2.RANSAC)
        
        img3 = cv2.warpPerspective(self.images[1],H,(canvas.shape[1],canvas.shape[0]))        
        canvas = self.image_stitch(canvas,img3)
        can_r, can_c = canvas.shape[0], canvas.shape[1]

        if self.save_path is not None:  
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
                print(f'Folder created at {self.save_path}')
            
            folder = self.image_folder.split('/')[-1]   
            save_img_name = os.path.join(self.save_path,f'{folder}_panaroma.jpg')      
            cv2.imwrite(save_img_name, canvas)
            print(f'Image Saved at {save_img_name}')
        
        print('Panaroma Created!! ')
        cv2.imshow('Uncropped Panorama Canvas', canvas)

        # Start from bottom and keep looking for zeros from bottom and crop. 
        y_bottom = can_r-padding 
        for y_val in range(can_r-1,can_r-padding, -1):
            if not (canvas[y_val, 0] == [0,0,0]).all():
                y_bottom = y_val
                break
        for x_val in range(can_c-1,0, -1):
            if not (canvas[0,x_val] == [0,0,0]).all():
                x_bottom = x_val
                break
        canvas_crop = canvas[0:y_val, 0:x_val]
        # cv2.imshow('panorama',canvas_crop)
        
        cv2.waitKey(0)


if __name__ == '__main__':

    images_dir = 'test_images/panorama_img/set1'
    save_dir = 'test_images/panorama_img/results'

    pan = Panorama(images_dir, save_path=save_dir, show_imgs=True)
    pan.createPanaroma()

    # img1 = cv2.imread('panorama_img/set1/1.jpeg',0)
    # img2 = cv2.imread('panorama_img/set1/2.jpeg',0)
    # pan.findSIFTfeatures(img1,img2, show_match=True)
    