'''
Correlation coeficient between two vectors using the dot product.
Note that the code use explicit 'for' statements to go through
the vector values. We do that to keep the code as simple as we can
for Numba/Cuda pre compilation. (pylint : disable=consider-using-enumerate)
Also Note the thrid parameter 'coef'. This is also just for numba/cuda
compatibility.
'''

# Importing external libs
import math

def corrcoef(vector01, vector02, coef=0):
  '''Correlation coeficient between two vectors using dot product'''
  max_v1, max_v2 = (0, 0)
  for i in range(len(vector01)): # pylint: disable=consider-using-enumerate
    # looking the max value in each vector
    max_v1 = vector01[i] if max_v1 < vector01[i] else max_v1
    max_v2 = vector02[i] if max_v2 < vector02[i] else max_v2

  v1_magnitude, v2_magnitude = (0, 0)
  for i in range(len(vector01)): # pylint: disable=consider-using-enumerate
    # normalize the vector value
    norm_v1 = vector01[i] / max_v1
    norm_v2 = vector02[i] / max_v2

    # calculating the dot product
    coef += norm_v1 * norm_v2

    # sum the power of the norm in order
    # to calculate the magnitude of each vector
    v1_magnitude += math.pow(norm_v1, 2)
    v2_magnitude += math.pow(norm_v2, 2)

  v1_magnitude = math.sqrt(v1_magnitude)
  v2_magnitude = math.sqrt(v2_magnitude)
  return coef / (v1_magnitude * v2_magnitude)
