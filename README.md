# Panorama Stitching from Scratch

Goal of this repo is to create a horizontal Panorama from a set of images. 

## Tasks

### For Harris Corner Detection

- [X] Basic Harris Detection Code
- [ ] Implement Non Max Supression 

### For Panorama Stitching

- [x] Implement Basic Image stitching for any number of images using OpenCV functions
  - [x] `Features`: Inbuilt SIFT features and SIFT descriptors 
  - [x] `Homography`: RANSAC (inbuilt)
  - [ ] `Blending` : Weighted transformed images
  
- [ ] Implement Basic Image stitching for any number of images from scratch
  - [ ] `Features`: Harris Corner features and SIFT descriptors of `HarrisCornerDetect.py`
  - [ ] `Homography`: RANSAC (from scratch)
  - [ ] `Blending` : Weighted transformed images
