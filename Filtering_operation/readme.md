## First of all download the following packages
*pip install PIL*
*pip install numpy*
## Run the following command in terminal to get blur images
python blur.py --sigma 2.0 ./princeton_small.jpg ./blur_princeton.jpg
## Run the following command in terminal to get sharpen images
python sharpen.py ./princeton_small.jpg ./sharpen_princeton.jpg
## Run the following command in terminal to detect the edge in image
python edge_detection.py ./princeton_small.jpg ./edge2.jpg
