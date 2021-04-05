# Panorama Stitching from Scratch

Goal of this repo is to create a horizontal Panorama from a set of images. 

## Tasks

### For Harris Corner Detection

- [X] Basic Harris Detection Code
- [x] Implement Non Max Supression 

### For Panorama Stitching

- [x] Implement Basic Image stitching for any number of images using OpenCV functions
  - [x] `Features`: Inbuilt SIFT features + SIFT descriptors 
  - [x] `Homography`: RANSAC (inbuilt)
  - [ ] `Blending`: Weighted transformed images
  - [ ] `Features`: Compare with Harris features and SIFT features
- [ ] Crop the final panorama image
- [ ] Try to make a better canvas length and breadth

## Contributers

- **Rohit Lal** - [Website](http://take2rohit.github.io/)
- **Khush Agrawal** - [Website](https://khush3.github.io/)

## References

- https://www.youtube.com/watch?v=J1DwQzab6Jg&list=PL2zRqk16wsdp8KbDfHKvPYNGF2L-zQASc
<!-- - https://stackoverflow.com/questions/10632617/how-to-remove-black-part-from-the-image -->
