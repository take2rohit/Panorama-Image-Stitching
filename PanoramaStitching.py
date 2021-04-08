import cv2
import numpy as np
import os, glob
from HarrisCornerDetect import HarrisDetector

class Panorama():

    def __init__(self, image_folder, show_imgs=False, random_order=False,
                 feature_selector='HARRIS', y_originshift=0,bottom_pad=100):

        '''
        Parameters:

            image_folder : Folder consisting of many images to be 
                            be stitched horizontally

            save_path : Directory to save panorama image. If directory 
                        doesnt exists, code will create automatically

            show_imgs : Show input images which are to be stitches

            y_originshift : Shift first reference image of panorama 
                            to particular y value in canvas

            bottom_pad : Add black padding to the bottom of canvas 

            random_order : Specify if the images are random or sequential
                           (Left to Right) or (Right to Left)

            feature_selector : Choose keypoint from SIFT or HARRIS 
        '''
        
        self.x_originshift, self.y_originshift = 0, y_originshift
        self.bottom_pad = bottom_pad
        self.feature_selector = feature_selector
        self.image_folder = image_folder
        img_path = sorted(glob.glob(os.path.join(self.image_folder,'*')))
        self.images = [cv2.imread(p) for p in img_path]
        self.bw_images = [cv2.imread(p,0) for p in img_path]
        
        if random_order:
            print('Computing image sequence...')
            self.order = self.find_best_matches()
            self.images = [self.images[i] for i in self.order]
            self.bw_images = [self.bw_images[i] for i in self.order]
            print('Completed Finding image sequence!')

        if show_imgs:
            for c, img in enumerate(self.images):
                cv2.namedWindow(str(c),cv2.WINDOW_NORMAL)
                cv2.imshow(str(c), img)
            print('\nShowing a set of all images')
            print('Press any key to create a Panorama',end='\n \n')
            cv2.waitKey(0)

    def save_image_fn(self, save_path, canvas):
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f'Folder created at {save_path}')
        
        folder = self.image_folder.split('/')[-1]   
        save_img_name = os.path.join(save_path,f'{folder}_panaroma.jpg')      
        cv2.imwrite(save_img_name, canvas)
        print(f'Image Saved at {save_img_name}')

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
        pairs_num_correspondences = []
        for i in range(len(self.images)):
            best_idx = -1
            best_idx_nums = 0
            second_best_idx = -1
            second_best_idx_nums = 0
            for j in range(len(self.images)):
                if j == i:
                    continue
                num_matches = self.find_num_matches(self.bw_images[i], self.bw_images[j])
                if num_matches >= second_best_idx_nums:
                    if num_matches >= best_idx_nums:
                        second_best_idx_nums = best_idx_nums
                        second_best_idx = best_idx
                        best_idx_nums = num_matches
                        best_idx = j
                    else:
                        second_best_idx_nums = num_matches
                        second_best_idx = j
            pairs.append([i, best_idx])
            pairs_num_correspondences.append(best_idx_nums)
            pairs.append([i, second_best_idx])
            pairs_num_correspondences.append(second_best_idx_nums)
        worst_idx = np.argmin(pairs_num_correspondences)
        del pairs_num_correspondences[worst_idx]
        del pairs[worst_idx]
        worst_idx = np.argmin(pairs_num_correspondences)
        del pairs_num_correspondences[worst_idx]
        del pairs[worst_idx]
        link = self.find_link(pairs)
        return link

    def find_link(self, connections):
        def op(ip):
            if ip == 0:
                return 1
            return 0
        c = connections.copy()
        link = [c[0][0], c[0][1]]
        del c[0]
        while len(link) < int((len(connections)+2)/2):
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
        width *= 2 
        
        cnv_shp = (img1_shp[0]+bottom_pad,width,img1_shp[2])

        self.x_originshift =  width//2

        canvas = np.zeros(cnv_shp,np.uint8) 
        canvas_mask = np.zeros_like(canvas)

        if self.y_originshift+img1_shp[0] > canvas.shape[0]:
            raise Exception(f"x_originshift should be less than {canvas.shape[0]-img1_shp[0]}")

        if self.x_originshift+img1_shp[1] > canvas.shape[1]:
            raise Exception(f"y_originshift should be less than {canvas.shape[1]-img1_shp[1]}")


        canvas[self.y_originshift:self.y_originshift+img1_shp[0],
            self.x_originshift:self.x_originshift+img1_shp[1]] = self.images[0]
        
        canvas_mask[self.y_originshift:self.y_originshift+img1_shp[0],
            self.x_originshift:self.x_originshift+img1_shp[1]] = 255

        return canvas,canvas_mask

    def image_blend(self, canvas, img2,canvas_mask):

        graycanvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
        grayImg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        flg = np.array((graycanvas,grayImg2))
        idx = np.argmax(flg,axis=0)
        id1 = np.where(idx ==0)
        id2 = np.where(idx==1)
        img = np.zeros_like(canvas)
        img[id1] = canvas[id1]
        img[id2] = img2[id2]
        
        canvas_mask[id2] = 255       

        return  img,canvas_mask

    def findSIFTfeatures(self, img1, img2, top_n=25, show_match=False):

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

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
  
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
            cv2.namedWindow('SIFT matched images', cv2.WINDOW_NORMAL)
            cv2.imshow('SIFT matched images', match_pt_img)
            cv2.waitKey(0)

        return (kp1, kp2), best_n

    def crop_canvas(self, canvas_mask, canvas):
        # cv2.imshow('ncv' , canvas_mask)

        canvas_gr = cv2.cvtColor(canvas_mask,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(canvas_gr,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

        # cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,0),2)
        max_visibile_im_canvas = canvas[y:y+h, x:x+w, :]

        return max_visibile_im_canvas

    def sequential_matmal(self,H_vals):
        
        H = H_vals[0]
        
        if len(H_vals) > 1 :
            for i in range(len(H_vals)-1):
                H = np.matmul(H,H_vals[i+1])
        
        return H
        
    def createPanaroma(self, show_output=False, save_path=None):

        canvas, canvas_mask = self.createCanvas(self.bottom_pad)

        
        img3 = self.bw_images[0].copy()
        T = np.array(   [[ 1 , 0 , self.x_originshift],
                        [ 0 , 1 , self.y_originshift],
                        [ 0 , 0 , 1]])
        H_vals = []

        for i in range(len(self.images)-1):
            img1 = img3
            
            if self.feature_selector == "SIFT":
                (kp1, kp2), best_n = self.findSIFTfeatures(self.bw_images[i],self.bw_images[i+1])
            
            if self.feature_selector == "HARRIS":
                (kp1, kp2), best_n = self.findHarrisfeatures(self.images[i],self.images[i+1],show_match=False)

            img1_pts = np.float32([kp1[val.queryIdx].pt for val in best_n]).reshape(-1,1,2)
            img2_pts = np.float32([kp2[val.trainIdx].pt for val in best_n]).reshape(-1,1,2)

            H,mask = cv2.findHomography(img2_pts,img1_pts,cv2.RANSAC)
            H = np.array(H)
            H_vals.append(H)
            
            H = self.sequential_matmal(H_vals)
            
            translatedH = np.matmul(T,H)
            
            # T = np.eye(3)
            img3 = cv2.warpPerspective(self.images[i+1],translatedH,
                                        (canvas.shape[1],canvas.shape[0])   )    

            canvas, canvas_mask = self.image_blend(canvas,img3,canvas_mask)
            # cv2.namedWindow('canvas',cv2.WINDOW_NORMAL)
            # cv2.imshow('canvas', canvas)
            # cv2.waitKey(0)
        max_im_canvas = self.crop_canvas(canvas_mask,canvas)

        if save_path is not None:  
            self.save_image_fn(save_path, max_im_canvas)
        
        if show_output:
            cv2.namedWindow('Normal Rectangle Panorama', cv2.WINDOW_NORMAL)
            cv2.imshow('Normal Rectangle Panorama', max_im_canvas)
            
        cv2.waitKey(0)
 
    def findHarrisfeatures(self, img1, img2, top_n=25, thresh_ratio = 0.01, show_match=False):

        '''
        Find good features to match using SIFT

        Parameters:
            img1 : First numpy image
            img2 : Second numpy image
            top_n : State how many top n features needed (default - 25)
            thresh_ratio : Higher thresh ratio more Harris corners
            show_match : Display image of matching
        
        Returns
            (kp1, kp2) : Keypoints of img1 and img2 respectively
            best_n : list of best n keypoints (default - 25)
        '''

        
        Harris_img1 = HarrisDetector(img1)
        kp1 = Harris_img1.findCorners(thresh_ratio=thresh_ratio, nms_threshold=20)

        Harris_img2 = HarrisDetector(img2)
        kp2 = Harris_img2.findCorners(thresh_ratio=thresh_ratio, nms_threshold=20)

        sift = cv2.SIFT_create()
        kp1,des1 = sift.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1 )
        kp2,des2 = sift.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2 )
        
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
            cv2.namedWindow('SIFT matched images', cv2.WINDOW_NORMAL)
            cv2.imshow('SIFT matched images', match_pt_img)
            cv2.waitKey(0)

        return (kp1, kp2), best_n

if __name__ == '__main__':

    ############# Run for a single set #############

    root = 'images/panorama_img/set1'
    save_dir = 'images/stitched_results'


    pan = Panorama(root, show_imgs=False, random_order=True, 
                        feature_selector='HARRIS')

    # pan.findHarrisfeatures(pan.images[0],pan.images[1])
    pan.createPanaroma(show_output=True , save_path=None)

    ############### Run for all sets ###############

    # root = 'images/panorama_img/'
    # save_dir = 'images/stitched_results'

    # for image_folder in os.listdir(root):
    #     print(f'\nCurrently testing {image_folder}')
    #     pan = Panorama(os.path.join(root,image_folder), show_imgs=False,
    #                 y_originshift=250, bottom_pad=900, random_order=True)

    #     pan.createPanaroma(show_output=False, save_path=save_dir)

    