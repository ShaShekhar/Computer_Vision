import cv2
import numpy as np

# Read the images
foreground = cv2.imread('comp_foreground.jpg')
background = cv2.imread('comp_background.jpg')
alpha = cv2.imread('comp_mask.jpg')
# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensities 0 and 1
alpha = alpha.astype(float)/255
# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha,foreground)
# Multiply the background with the alpha matte
background = cv2.multiply(1.0 - alpha, background)
# Add the masked foreground and background
out_image = cv2.add(foreground, background).astype(np.uint8)
# Display image
cv2.imshow('out_image', out_image)
#cv2.imwrite('composite.jpg',out_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
