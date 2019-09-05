import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
sess = tf.Session()

# First readimage,convert its dtype from uint8 to float32 and convert rgb to grayscale then resize and squeeze its dim.
img = imread("china.jpg")
img = img.astype(np.float32)
img = tf.image.rgb_to_grayscale(img) #print(img.get_shape())-->(1944, 2592, 1)
img = tf.image.resize_images(img,[512,512],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
img = tf.squeeze(img) # (512, 512)
img_4d = tf.reshape(img, [1,img.get_shape().as_list()[0], img.get_shape().as_list()[1], 1])
#img_4d = tf.reshape(img,[1,img.shape[0],img.shape[1],1])

# Create Gaussian Kernal
x = tf.linspace(-3.0, 3.0, 16)
mean = 0.0
sigma = 1.0
z = tf.exp(tf.negative(tf.pow((x-mean),2)/2*sigma**2))*(1/tf.sqrt(2*3.1415*sigma**2))
ksize = z.get_shape().as_list()[0]
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

# Create 2d sin wave
xs = tf.linspace(-3.0, 3.0, 16)
ys = tf.sin(xs)
s_size = ys.get_shape().as_list()[0]
ones = tf.ones((1, 16))
ys_2d = tf.matmul(tf.reshape(ys, [s_size, 1]), ones)

# Gabor kernal
gabor = tf.multiply(z_2d, ys_2d)
gabor_4d = tf.reshape(gabor,[gabor.get_shape().as_list()[0], gabor.get_shape().as_list()[0], 1, 1])

# convolved the Gabor kernal on the image
convolved = tf.nn.conv2d(img_4d, gabor_4d, strides=[1,1,1,1], padding='SAME')
convolved = tf.squeeze(convolved)

plt.imshow(convolved.eval(session=sess), cmap='gray')
plt.show()
