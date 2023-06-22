import numpy as np
import cv2
import time

# Read the image
image = cv2.imread(r'C:\Users\Shehab\Downloads\lena.png')

# Get the kernel size from user input
kernel_size = int(input("Enter the kernel size (odd value): "))

# Validate the kernel size
if kernel_size % 2 == 0:
    kernel_size += 1

start = time.time()
# Define the kernel for blurring with all weights equal to 1/size^2
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

# Apply the convolution operation with zero padding (borderType=cv2.BORDER_CONSTANT)
blurred_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
end = time.time()
print("Time taken by the Sequential approach is: ", end - start, " seconds.")

# Display the original and blurred images
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
