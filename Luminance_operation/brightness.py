# import OpenCV for reading the image from disk
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--brightness', nargs='?', type=float, default=1.0)
parser.add_argument('input_file', help='The input image file.')
args = parser.parse_args()

# Read the image in form of RGB
img = cv2.imread(args.input_file,1)
# Get its height, width and channel
height = img.shape[0]
width = img.shape[1]
channel = img.shape[2]

brightness_factor = args.brightness
# Loop through each pixel in a image and scale by a real value factor
for i in range(height):
    for j in range(width):
        for k in range(channel):
            a = img.item(i,j,k) # Here a is integer value
            b = a*brightness_factor  # After multiplying float it becomes float
            if b > 255:
                b = 255
            # Since the type of image is uint8, Assigning it float value it automatically convert it to uint8
            img.itemset((i,j,k),b)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.imwrite('image_bright.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
