'''
Here we have the cuda kernels that we will call from our main script.
Note that we have a lot of pylint disable. That's because the not
supported cuda decorators.
'''

# Importing external libs
import numpy as np
from numba import cuda, uint8, float32

# Importing app lib
import corrcoef as my

# Note the hardcoded 4 value.
# Four is the number of item per pixel.
# They represent a pixel in the form of (r, g, b, alpha)
PIXEL_DATA_LENGTH = 4

# pre compile the standard corrcoef method to be use as cuda call
cuda_corrcoef = cuda.jit(restype=float32, argtypes=[float32[:], float32[:], float32], device=True)(my.corrcoef) # pylint: disable=bad-whitespace

@cuda.jit(argtypes=[float32[:,:,:], float32[:,:,:], float32[:,:]]) # pylint: disable=bad-whitespace
def correlate_frames_kernel(cuda_img01_frames, cuda_img02_frames, cuda_coeficients):
  '''To calculate the correlation between each frame'''
  coord_x = cuda.blockIdx.x
  coord_y = cuda.blockIdx.y
  coef = 0
  cuda_coeficients[coord_x][coord_y] = cuda_corrcoef(
    cuda_img01_frames[coord_x][coord_y],
    cuda_img02_frames[coord_x][coord_y],
    coef,
  )

@cuda.jit(argtypes=[uint8[:,:,:], float32[:,:,:], uint8]) # pylint: disable=bad-whitespace
def flatten_frame_kernel(cuda_img, cuda_flatten_frame, frame_size):
  '''To flatten each window frame from the img array to a vector'''
  index_x, index_y = cuda.grid(2) # pylint: disable=not-callable
  coord_x = cuda.blockIdx.x
  coord_y = cuda.blockIdx.y

  # Placing the frame data in a vector
  index = (cuda.threadIdx.x * frame_size + cuda.threadIdx.y) * PIXEL_DATA_LENGTH
  for pixel_data_index in range(PIXEL_DATA_LENGTH):
    cuda_flatten_frame[coord_x][coord_y][index] = cuda_img[index_y][index_x][pixel_data_index]
    index += 1

def pre_process(input_img, frames_per_height, frames_per_width, frame_size):
  '''Helper to flatten image frames'''
  flatten_length = frame_size * frame_size * PIXEL_DATA_LENGTH

  # set grid and block
  threadsperblock = (frame_size, frame_size)
  blockspergrid = (frames_per_width, frames_per_height)

  # flatten frames
  framed_img = np.zeros((frames_per_width, frames_per_height, flatten_length), dtype=np.float32)
  cuda_img = cuda.to_device(input_img)
  cuda_framed_img = cuda.to_device(framed_img)
  flatten_frame_kernel[blockspergrid, threadsperblock](cuda_img, cuda_framed_img, frame_size) # pylint: disable=unsubscriptable-object
  cuda_framed_img.to_host()
  return framed_img
