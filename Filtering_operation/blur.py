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

def gaussian_kernel_2d(sigma):
    kernel_radius = np.ceil(sigma)*3
    kernel_size = kernel_radius * 2 + 1 # This is the formula for kernel_size
    ax = np.arange(-kernel_radius, kernel_radius+1., dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax) # Use meshgrid to represent kernel in cartesian co-ordinate.
    kernel = np.exp(-(xx**2 + yy**2) / (2.*sigma**2)) # Apply Gaussian formula
    return kernel/np.sum(kernel) # Normalize the kernel so we don't need to do during convolution operation.

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
                output[depth,r,c] = np.sum(image_pad[depth, r:r+kh, c:c+kw] * kernel[:,:])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', nargs='?', type=float, default=2.0)
    parser.add_argument('input_file', help='The input image file.')
    #parser.add_argument('output_file', help='The output image file.')
    args = parser.parse_args()
    # Read input image from disk
    img = read_image(args.input_file)
    # create output as same size of input filled with zeros.
    output = np.zeros(img.shape, dtype=np.float32)
    # Generate Gaussian kernel according to the value of sigma
    kernel_2d = gaussian_kernel_2d(args.sigma)
    # Convolve the Gaussian kernel on image
    output = convolve(img, output, kernel_2d)
    # Write the convolved image to disk
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
