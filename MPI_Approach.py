import numpy as np
import cv2
import time
from mpi4py import MPI

# Initialize MPI communicator, rank, and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Master process reads the image
if rank == 0:
    image_path = r'C:\Users\Shehab\Downloads\lena.png'
    image = cv2.imread(image_path)
    image_height = image.shape[0]
    # Calculate the number of rows to process per process
    rows_per_process = image_height // size
    remainder_rows = image_height % size
else:
    # Other processes initialize variables as None
    image = None
    rows_per_process = None
    remainder_rows = None

start = time.time()
# Broadcast image, rows_per_process, and remainder_rows to all processes
image = comm.bcast(image, root=0)
rows_per_process = comm.bcast(rows_per_process, root=0)
remainder_rows = comm.bcast(remainder_rows, root=0)

# Calculate the starting and ending rows for each process
start_row = rank * rows_per_process + 1
end_row = start_row + rows_per_process + 1

# Adjust the last process to handle the remainder rows if any
if rank == size - 1:
    end_row += remainder_rows

# Each process extracts its portion of rows from the image
local_rows = image[start_row:end_row]

# Define the kernel for blurring
kernel = np.ones((3, 3), dtype=np.float32) / 9

# -1 is passed as the second argument, indicating that the output image will have the same depth as the input image
# The kernel is applied to each pixel in the image, producing a weighted sum of the pixel values in the neighborhood defined by the kernel
# This operation blurs the image by smoothing out pixel intensities
# The border type is set to cv2.BORDER_REPLICATE, which replicates the border pixels when applying the convolution near the image boundaries
local_blurred_rows = cv2.filter2D(local_rows, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# Gather the processed rows from all processes using MPI Gather
gathered_blurred_rows = comm.gather(local_blurred_rows, root=0)

# Master process collects all the processed rows and constructs the final blurred image
if rank == 0:
    blurred_image = np.concatenate(gathered_blurred_rows, axis=0)
    end = time.time()
    print("Total time taken by MPI is: ", end-start, " seconds.")
    cv2.imshow("Original Image", image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)
MPI.Finalize()
