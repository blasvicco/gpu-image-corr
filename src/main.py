#!/usr/bin/env python
'''
IMPORTANT: This script require CUDA libs!!!
!apt-get install nvidia-cuda-toolkit
!export NUMBAPRO_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice
!export NUMBAPRO_NVVM=/usr/lib/x86_64-linux-gnu/libnvvm.so

If you don't want to add cuda libs in your local or you don't have
a GPU, then try 'https://colab.research.google.com/'

Python script to benchmark images processing using
CPU versus GPU (parallelization)

The script will compare two images with the same size
and detect the differences between them using framed
window correlations.

It will generate an image per method with the two images
one at the side of the other.
The difference will be marked with a red border in the
frames that doesn't reach the threshold.

Available script arguments:
ARGUMENTS                  : TYPE                                                         : DEFAULT VALUE
-img1 / --image-01         : string (path to one image)                                   : data/img01.png
-img2 / --image-02         : string (path to the other image)                             : data/img02.png
-f    / --frame-size'      : integer (size of the frame / window NxN)                     : 4
-t    / --threshold        : float (threshold used to mark frame as different [0.0, 1.0]) : 0.8
-coi  / --cpu-output-img   : string (cpu output image path / filename)                    : data/cpu_output_img.png
-goi  / --gpu-output-img   : string (gpu output image path / filename)                    : data/gpu_output_img.png

'''

# Importing external libs
import argparse
from timeit import default_timer as timer
from numpy import append, asarray, float32, zeros
from numba import cuda
from PIL import Image, ImageDraw


# Importing app libs
import corrcoef as my
import cuda_kernels as ck

def create_comparison_image(img01, img02, coeficients, threshold, path, frame_size): # pylint: disable=too-many-arguments
  '''Create a comparison image between img01 and img02'''

  # image parametrization
  frames_per_height = int(img01.size[1] / frame_size)
  frames_per_width = int(img01.size[0] / frame_size)

  # creating a drawable image from img02
  tmp_img = Image.new('RGBA', img02.size, (255, 255, 255, 0))
  tmp_img.paste(img02)

  # drawing frames where difference is detected
  detected_diff = ImageDraw.Draw(tmp_img)
  for coord_y in range(frames_per_height):
    frame_from_y = coord_y * frame_size
    frame_to_y = coord_y * frame_size + frame_size
    for coord_x in range(frames_per_width):
      frame_from_x = coord_x * frame_size
      frame_to_x = coord_x * frame_size + frame_size
      if coeficients[coord_x][coord_y] < threshold:
        detected_diff.rectangle(
          [(frame_from_x, frame_from_y), (frame_to_x, frame_to_y)],
          fill=None,
          outline='red'
        )

  # joining images
  output = Image.new(
    'RGBA',
    (img02.size[0] * 2, img02.size[1]),
    (255, 255, 255, 0)
  )
  output.paste(img01)
  output.paste(tmp_img, (img02.size[0], 0))
  output.save(path)

def cpu_find_difference(img01, img02, frame_size):
  '''Find the difference between 2 images using the CPU'''
  print('Comparing the images using the CPU.')
  start = timer() # initializing timer

  # image parametrization
  frames_per_height = int(img01.shape[0] / frame_size)
  frames_per_width = int(img01.shape[1] / frame_size)
  coeficients = zeros((frames_per_width, frames_per_height), dtype=float32)

  # calculate the correlation between the images frame by frame
  for coord_y in range(frames_per_height):
    frame_from_y = coord_y * frame_size
    frame_to_y = coord_y * frame_size + frame_size
    framed_img01_y = img01[frame_from_y:frame_to_y]
    framed_img02_y = img02[frame_from_y:frame_to_y]
    for coord_x in range(frames_per_width):
      frame_from_x = coord_x * frame_size
      frame_to_x = coord_x * frame_size + frame_size
      for row in range(frame_size):
        framed_img01_yx = append(
          framed_img01_yx,
          framed_img01_y[row][frame_from_x:frame_to_x].flatten()
        )
        framed_img02_yx = append(
          framed_img02_yx,
          framed_img02_y[row][frame_from_x:frame_to_x].flatten()
        )

      coeficients[coord_x][coord_y] = my.corrcoef(framed_img01_yx, framed_img02_yx)

  # printing total time needed to find the difference
  delta_time = timer() - start
  print('Difference detected in {0:.2f}s.'.format(delta_time))
  return coeficients

def gpu_find_difference(img01, img02, frame_size):
  '''Find the difference between 2 images using the GPU'''
  print('Comparing the images using the GPU.')
  start = timer()  # initializing timer

  # image parametrization
  frames_per_height = int(img01.shape[0] / frame_size)
  frames_per_width = int(img01.shape[1] / frame_size)
  coeficients = zeros((frames_per_width, frames_per_height), dtype=float32)

  # flatten frames
  img01_frames = ck.pre_process(img01, frames_per_height, frames_per_width, frame_size)
  img02_frames = ck.pre_process(img02, frames_per_height, frames_per_width, frame_size)

  # set grid and block for corrcoef
  threadsperblock = (1, 1)
  blockspergrid = (frames_per_width, frames_per_height)

  # calculate the correlation between the images frame by frame
  cuda_img01_frames = cuda.to_device(img01_frames)
  cuda_img02_frames = cuda.to_device(img02_frames)
  cuda_coeficients = cuda.to_device(coeficients)
  ck.correlate_frames_kernel[blockspergrid, threadsperblock]( # pylint: disable=unsubscriptable-object
    cuda_img01_frames, cuda_img02_frames, cuda_coeficients
  )
  cuda_coeficients.to_host()

  # printing total time needed to find the difference
  delta_time = timer() - start
  print('Diff detected in {0:.2f}s.'.format(delta_time))
  return coeficients

def main():
  '''Main script'''

  # read arguments
  args = read_args()

  # load the image
  img01 = Image.open(args.image_01)
  img02 = Image.open(args.image_02)

  # convert images to numpy array
  img01_data = asarray(img01)
  img02_data = asarray(img02)

  # finding difference using the cpu
  cpu_corrcoef = cpu_find_difference(img01_data, img02_data, args.frame_size)
  create_comparison_image(
    img01, img02, cpu_corrcoef, args.threshold, args.cpu_output_img, args.frame_size
  )

  # finding difference using the gpu
  gpu_corrcoef = gpu_find_difference(img01_data, img02_data, args.frame_size)
  create_comparison_image(
    img01, img02, gpu_corrcoef, args.threshold, args.gpu_output_img, args.frame_size
  )


def read_args():
  '''Read arguments function'''
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-img1', '--image-01', type=str)
  parser.add_argument('-img2', '--image-02', type=str)
  parser.add_argument('-f', '--frame-size', type=int)
  parser.add_argument('-t', '--threshold', type=float)
  parser.add_argument('-coi', '--cpu-output-img', type=str)
  parser.add_argument('-goi', '--gpu-output-img', type=str)

  parser.set_defaults(
    image_01='data/img01.png',
    image_02='data/img02.png',
    frame_size=4,
    threshold=0.8,
    cpu_output_img='data/cpu_output_img.png',
    gpu_output_img='data/gpu_output_img.png',
  )

  return parser.parse_args()

if __name__ == '__main__':
  main()
