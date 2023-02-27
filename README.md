[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# FaceSwap
## Classical Approach
This repository contains code to implement an end-to-end pipeline for swapping faces from images and videos using classical and deep learning approach.
### Pipeline
<img src="https://github.com/naitri/FaceSwap/blob/main/git_images/pipl.png"  align="center" alt="Undistorted" width="400"/>

### Facial Landmarks detection
Inbuilt library dlib is used which is based on SVM face detector.

<img src="https://github.com/naitri/FaceSwap/blob/main/git_images/detect.png"  align="center" alt="Undistorted" width="400"/>
 
### Face Warping using Triangulation and Thin Plate Spline
<img src="https://github.com/naitri/FaceSwap/blob/main/git_images/tps.png"  align="center" alt="Undistorted" width="400"/>
  
Instructions to run FaceSwap using Traditional approach (triangulation & Thin Plate Spline):
1. Set input1 and input 2 as the required source and destination paths in Wrapper.py main. Note:
 	- If the two inputs are the same video, set input1 as video path and input2 as None
 	- If one input is a video and the other an image, set input1 as video path and input2 as image path
2. Set method as "TRI" for triangulation or "TPS" for Thin Plate Spline algorithms respectively.
3. Run Wrapper.py using `python3 Wrapper.py`

Instructions to run PRNet:
1. Download the model weight from https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view
2. Place the file in ./PRNet/Data/net-data
3. Create a conda environment or venv with python2.7, tf 1.13 (gpu version), cv2 4.2.1, numpy and dlib.
4. In prnetWrapper.py, specify path to the source image/video and path to the destination image/video.
5. Run in command line: `python prnetWrapper.py`.

## Results

### Facial Landmarks
<p float="left">
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data//outputs/dst_landmarks.jpg" width="350" /> 
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data/outputs/src_landmarks.jpg" width="350" />
</p>

### Delaunay's Triangulation
<p float="left">
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data//outputs/triangles_dst.jpg" width="350" /> 
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data/outputs/triangles_src.jpg" width="350" />
</p>

### Final Swap
<p float="center">
  	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data/outputs/img_swap.jpg" width="350"/>	
</p>

### Swap in the same frame
<p float="left">
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data/two_people.jpg" width="350" /> 
	<img src="https://github.com/niteshjha08/FaceSwap/blob/main/data/outputs/frame_swap.jpg" width="350" />
</p>
