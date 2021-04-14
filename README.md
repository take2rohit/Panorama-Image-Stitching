# Panoramaic Image Stitching Library

Goal of this repo is to creates a Panorama from a set of images. The images may not be in order of apperences. This repo also contains code of Harris corner written from scratch. It compares SIFT and Harris features for image stitching. A report of this project can be found at [Report-Panorama.pdf](Report-Panorama.pdf). Feel free to raise issue or contact the contributors in case of any doubts or development of this code. 

*Note: this code will not work for cylindrical panoramas*

## How to use

- Clone the repository by typing and Navigate to cloned directory
- Install all the dependencies   
  ```bash
  git clone https://github.com/take2rohit/Panorama-Image-Stitching.git
  cd Panorama-Image-Stitching
  pip3 install -r requirements.txt
  ```
- Open [PanoramaStitching.py](PanoramaStitching.py) 
  - Change variable `root` to specify location of directory containg piecewise panoramaic images
  - Change `save_dir` to save the results of stitched panorama

## Using this repo as library

This is an extremely low code (4 lines) modular library to create panoramic images from a set of images. Just clone the repository in working code directory and run the below python code following commands

```python
from PanoramaStitching import Panorama

root = 'images/panorama_img/set3'

pan = Panorama(root)
stitchedIm = pan.createPanaroma(show_output=True)

```

For finer controls please see the docstrings of the code. 

## Results

![](images/stitched_results/SIFT_set1_pan.jpg)

![](images/stitched_results/SIFT_set2_pan.jpg)

![](images/stitched_results/SIFT_set3_pan.jpg)

![](images/stitched_results/SIFT_set4_pan.jpg)

## Contributers

- **Rohit Lal** - [Website](http://take2rohit.github.io/)
- **Khush Agrawal** - [Website](https://khush3.github.io/)
