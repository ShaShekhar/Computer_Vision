# import nesessary library
from PIL import Image #for image load and save
import numpy as np # for matrix calculation
import argparse # parametrize the input
import cv2

def read_image(path):
    img = np.array(Image.open(path))
    # Change the image from (H,W,C) to (C,H,W) and convert into float
    float_image = np.ascontiguousarray(img.transpose(2,0,1),dtype=np.float32)
    float_image /= 255 # Normalize the float value
    return float_image

def write_image(img,path):
    # convert the float value to a range of 0 to 255
    img *= 255
    # Change the image from (C,H,W) to (H,W,C) and convert into unsinged 8 bit integer.
    img = img.transpose(1,2,0).astype(np.uint8)
    Image.fromarray(img).save(path)


def convolve(image, output, kernel):
    channel = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    kh = kernel.shape[0] #kernel_height
    kw = kernel.shape[1] #kernel_width

    # Do Convolution
    for depth in range(channel):
        for r in range(1,image_height-2):
            for c in range(1,image_width-2):
                output[depth,r,c] = np.sum(image[depth, r:r+kh, c:c+kw] * kernel[:,:])*(1.0/6.0)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The input image file.')
    #parser.add_argument('output_file', help='The output image file.')
    args = parser.parse_args()
    # Read input image from disk
    img = read_image(args.input_file)
    # create output as same size of input filled with zeros.
    output = np.zeros(img.shape, dtype=np.float32)
    # Horizontal edge operator
    Hx = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])
    # Vertical edge operator
    Hy = np.array([[-1,-2,-1],
                   [0 ,0 ,0],
                   [1 ,2, 1]])
    # Convolve horizontal edge operator on the input image to get edge in horizontal direction
    img_x = convolve(img,output, Hx)
    # Convolve vertical edge operator on the input image to get edge in vartical direction
    img_y = convolve(img,output, Hy)
    # Combine both horizontal and vertical edge to get edge in both direction.
    img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
    # Write the output image to disk
    #write_image(img_out, args.output_file)
    img_out *= 255
    # Change the image from (C,H,W) to (H,W,C) and convert into unsinged 8 bit integer.
    img_out = img_out.transpose(1,2,0).astype(np.uint8)
    img_out = img_out[:,:,::-1]
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
