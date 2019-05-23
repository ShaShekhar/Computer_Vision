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
    img *= 255 # convert the float value to a range of 0 to 255
    # Change the image from (C,H,W) to (H,W,C) and convert into unsinged 8 bit integer.
    img = img.transpose(1,2,0).astype(np.uint8)
    Image.fromarray(img).save(path)


def convolve(image, output, kernel):
    channel = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    kh = kernel.shape[0] #kernel_height
    kw = kernel.shape[1] #kernel_width
    # Pad the borders to perform convolution operation on whole image
    P = kh//2

    image_pad = np.pad(image,((0,0),(P,P),(P,P)),'constant',constant_values=0)
    # Do Convolution
    for depth in range(channel):
        for r in range(0,image_height):
            for c in range(0,image_width):
                output[depth,r,c] = np.sum(image_pad[depth, r:r+kh, c:c+kw] * kernel[:,:])/np.sum(kernel)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The input image file.')
    #parser.add_argument('output_file', help='The output image file.')
    args = parser.parse_args()
    # Read input image from disk
    img = read_image(args.input_file)
    # create output as same size of input filled with zeros.
    output = np.zeros(img.shape, dtype=np.float32)
    sharpen_filter = np.array([[-1,-1,-1],
                               [-1,-9,-1],
                               [-1,-1,-1]])
    # Convolve the sharpen filter on the image
    convolve(img, output,sharpen_filter)
    # Write the convolved image on the disk
    #write_image(output, args.output_file)
    output *= 255
    # Change the image from (C,H,W) to (H,W,C) and convert into unsinged 8 bit integer.
    output = output.transpose(1,2,0).astype(np.uint8)
    output = output[:,:,::-1]
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
