import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--contrast', nargs='?', type=float, default=1.0)
parser.add_argument('input_file', help='The input image file.')
args = parser.parse_args()

img = cv2.imread(args.input_file,1)
height, width, channel = img.shape[0],img.shape[1],img.shape[2]
gray_image = np.full((height, width, channel),100,dtype=np.uint8)
alpha = args.contrast
for i in range(height):
    for j in range(width):
        for k in range(channel):
            px = img.item(i,j,k)
            gray_px = gray_image[i,j,k]
            out_px = (1 - alpha)*gray_px + alpha*px
            if out_px > 255:
                out_px = 255
            img.itemset((i,j,k),out_px)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.imwrite('c_contrast_{}.jpg'.format(alpha),img)
cv2.waitKey(0)
cv2.destroyAllWindows()
