import cv2
import numpy as np
import os, glob

class Panorama():

    def __init__(self,image_folder, show_imgs=False,
                x_originshift=0, y_originshift=0,bottom_pad=100):

        '''
        Parameters:

            image_folder : Folder consisting of many images to be 
                            be stitched horizontally

            save_path : Directory to save panorama image. If directory 
                        doesnt exists, code will create automatically

            show_imgs : Show input images which are to be stitches

            x_originshift : Shift first reference image of panorama 
                            to particular x value in canvas

            y_originshift : Shift first reference image of panorama 
                            to particular y value in canvas

            bottom_pad : Add black padding to the bottom of canvas 
        '''
        
        self.x_originshift, self.y_originshift = x_originshift, y_originshift
        self.bottom_pad = bottom_pad
        self.image_folder = image_folder
        img_path = sorted(glob.glob(os.path.join(self.image_folder,'*')))
        self.images = [cv2.imread(p) for p in img_path]
        self.bw_images = [cv2.imread(p,0) for p in img_path]

        if show_imgs:
            for c, img in enumerate(self.images):
                cv2.namedWindow(str(c),cv2.WINDOW_NORMAL)
                cv2.imshow(str(c), img)
            print('\nShowing a set of all images')
            print('Press any key to create a Panorama',end='\n \n')
            cv2.waitKey(0)

    def createCanvas(self, bottom_pad):

        '''
        Creates a Canvas with first image appended in it. 
        Canvas dimension 
            height => height of first image + bottom_pad
            width => sum of width of all image

        '''

        img1_shp = self.images[0].shape
        width = 0
        for im in self.images:
            width += im.shape[1]

        cnv_shp = (img1_shp[0]+bottom_pad,width,img1_shp[2])
        canvas = np.zeros(cnv_shp,np.uint8) 

        # print(canvas.shape,(self.y_originshift+img1_shp[0],self.x_originshift+img1_shp[1] ))

        if self.y_originshift+img1_shp[0] > canvas.shape[0]:
            raise Exception(f"x_originshift should be less than {canvas.shape[0]-img1_shp[0]}")

        if self.x_originshift+img1_shp[1] > canvas.shape[1]:
            raise Exception(f"y_originshift should be less than {canvas.shape[1]-img1_shp[1]}")


        canvas[self.y_originshift:self.y_originshift+img1_shp[0],
            self.x_originshift:self.x_originshift+img1_shp[1]] = self.images[0]

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

    def createPanaroma(self,show_output=True, save_path=None):

        canvas = self.createCanvas(self.bottom_pad)
        img3 = self.bw_images[0].copy()
        T = np.array(   [[ 1 , 0 , self.x_originshift],
                        [ 0 , 1 , self.y_originshift],
                        [ 0 , 0 , 1]])
        for i in range(len(self.images)-1):
        # i = 0
            img1 = img3
            (kp1, kp2), best_n = self.findSIFTfeatures(img1,self.bw_images[i+1])

            img1_pts = np.float32([kp1[val.queryIdx].pt for val in best_n]).reshape(-1,1,2)
            img2_pts = np.float32([kp2[val.trainIdx].pt for val in best_n]).reshape(-1,1,2)
            
            H,mask = cv2.findHomography(img2_pts,img1_pts,cv2.RANSAC)
            
            translatedH = np.matmul(T,H)
            T = np.eye(3)
            img3 = cv2.warpPerspective(self.images[i+1],translatedH,
                                        (canvas.shape[1],canvas.shape[0])   )    
            canvas = self.image_stitch(canvas,img3)
        
        if save_path is not None:  
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                print(f'Folder created at {save_path}')
            
            folder = self.image_folder.split('/')[-1]   
            save_img_name = os.path.join(save_path,f'{folder}_panaroma.jpg')      
            cv2.imwrite(save_img_name, canvas)
            print(f'Image Saved at {save_img_name}')
        
        if show_output:
            cv2.namedWindow('Panorama Canvas', cv2.WINDOW_NORMAL)
            cv2.imshow('Panorama Canvas', canvas)

        # Start from bottom and keep looking for zeros from bottom and crop. 

        # can_r, can_c = canvas.shape[0], canvas.shape[1]
        # y_bottom = can_r-bottom_pad 
        # for y_val in range(can_r-1,can_r-bottom_pad, -1):
        #     if not (canvas[y_val, 0] == [0,0,0]).all():
        #         y_bottom = y_val
        #         break
        # for x_val in range(can_c-1,0, -1):
        #     if not (canvas[0,x_val] == [0,0,0]).all():
        #         x_bottom = x_val
        #         break
        # canvas_crop = canvas[0:y_val, 0:x_val]
        
        cv2.waitKey(0)


    def find_num_matches(self, img1, img2):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_features = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good_features.append(m)

        return len(good_features)


    def find_best_matches(self):
        pairs = []
        for i in range(len(self.images)):
            best_idx = -1
            best_idx_nums = 0
            for j in range(len(self.images)):
                if j == i:
                    continue
                num_matches = self.find_num_matches(self.bw_images[i], self.bw_images[j])
                if num_matches > best_idx_nums:
                    best_idx_nums = num_matches
                    best_idx = j
            pairs.append([i, best_idx])
        print('Original connections:', pairs)
        link = self.find_link(pairs)
        print('Link:', link)


    def find_link(self, connections):
        def op(ip):
            if ip == 0:
                return 1
            return 0
        c = connections.copy()
        link = [c[0][0], c[0][1]]
        del c[0]
        while len(link) < len(connections):
            leaf = link[0]
            x, y = np.where(np.array(c) == leaf)
            if len(x) > 0:
                x, y = x[0], y[0]
                element = c[x][op(y)]
                if element not in link:
                    link.insert(0, element)
                del c[x]
            leaf = link[-1]
            x, y = np.where(np.array(c) == leaf)
            if len(x) > 0:
                x, y = x[0], y[0]
                element = c[x][op(y)]
                if element not in link:
                    link.append(element)
                del c[x]
        return link


if __name__ == '__main__':

    ############# Run for a single set #############

    # root = 'test_images/panorama_img/set3'
    # save_dir = 'test_images/stitched_results'

    # pan = Panorama(root, show_imgs=False,
    #             x_originshift=500, y_originshift=0, bottom_pad = 300 )
    # pan.createPanaroma(show_output=True,save_path=None)

    ############### Run for all sets ###############

    root = 'images/panorama_img/'
    save_dir = 'images/stitched_results'

    for image_folder in os.listdir(root):
        print(f'\nCurrently testing {image_folder}')
        pan = Panorama(os.path.join(root,image_folder), show_imgs=False,
                    x_originshift=400, y_originshift=250, bottom_pad = 900 )
        pan.createPanaroma(show_output=False, save_path=save_dir)


    