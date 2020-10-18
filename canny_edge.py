# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from cupyx.scipy import signal
from PIL import Image
import argparse

# import functions
from helpers import interp2

"""# Tests and Visualization"""
def Test_script(I, E):
    test_pass = True

    # E should be 2D matrix
    if E.ndim != 2:
      print('ERROR: Incorrect Edge map dimension! \n')
      print(E.ndim)
      test_pass = False
    # end if

    # E should have same size with original image
    nr_I, nc_I = I.shape[0], I.shape[1]
    nr_E, nc_E = E.shape[0], E.shape[1]

    if nr_I != nr_E or nc_I != nc_E:
      print('ERROR: Edge map size has changed during operations! \n')
      test_pass = False
    # end if

    # E should be a binary matrix so that element should be either 1 or 0
    numEle = E.size
    numOnes, numZeros = E[E == 1].size, E[E == 0].size

    if numEle != (numOnes + numZeros):
      print('ERROR: Edge map is not binary one! \n')
      test_pass = False
    # end if

    return test_pass

'''
  Derivatives visualzation function
'''
def visDerivatives(I_gray, Mag, Magx, Magy):
    fig, (Ax0, Ax1, Ax2, Ax3) = plt.subplots(1, 4, figsize = (20, 8))

    Ax0.imshow(Mag, cmap='gray', interpolation='nearest')
    Ax0.axis('off')
    Ax0.set_title('Gradient Magnitude')

    Ax1.imshow(Magx, cmap='gray', interpolation='nearest')
    Ax1.axis('off')
    Ax1.set_title('Gradient Magnitude (x axis)')
    
    Ax2.imshow(Magy, cmap='gray', interpolation='nearest')
    Ax2.axis('off')
    Ax2.set_title('Gradient Magnitude (y axis)')

    # plot gradient orientation
    Mag_vec = Mag.transpose().reshape(1, Mag.shape[0] * Mag.shape[1]) 
    hist, bin_edge = cp.histogram(Mag_vec.transpose(), 100)

    ind_array = cp.array(cp.where( (cp.cumsum(hist).astype(float) / hist.sum()) < 0.95))
    thr = bin_edge[ind_array[0, -1]]

    ind_remove = cp.where(cp.abs(Mag) < thr)
    Magx[ind_remove] = 0
    Magy[ind_remove] = 0

    X, Y = cp.meshgrid(cp.arange(0, Mag.shape[1], 1), cp.arange(0, Mag.shape[0], 1))
    Ori = cp.arctan2(Magy, Magx)
    ori = Ax3.imshow(Ori, cmap='hsv')
    Ax3.axis('off')
    Ax3.set_title('Gradient Orientation')
    fig.colorbar(ori, ax=Ax3, )
    


'''
  Edge detection result visualization function
'''
def visCannyEdge(Im_raw, M, E):
    # plot image
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (12, 12))

    # plot original image
    ax0.imshow(Im_raw)
    ax0.axis("off")
    ax0.set_title('Raw image')

    # plot edge detection result
    ax1.imshow(M, cmap='gray', interpolation='nearest')
    ax1.axis("off")
    ax1.set_title('Non-Max Suppression Result')

    # plot original image
    ax2.imshow(E, cmap='gray', interpolation='nearest')
    ax2.axis("off") 
    ax2.set_title('Canny Edge Detection')

"""# Functions"""

'''
  Convert RGB image to gray one manually
  - Input I_rgb: 3-dimensional rgb image
  - Output I_gray: 2-dimensional grayscale image
'''
def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def findDerivatives(I_gray):
    '''
    File clarification:
        Compute gradient information of the input grayscale image
        - Input I_gray: H x W matrix as image
        - Output Mag: H x W matrix represents the magnitude of derivatives
        - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
        - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
        - Output Ori: H x W matrix represents the orientation of derivatives
    '''
    # TODO: complete function
    dx = cp.array([[1, 1, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
    dy = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)
    G = cp.array([[2, 4, 5, 4, 2],
                  [4, 9, 12, 9, 4],
                  [5, 12, 15, 12, 5],
                  [4, 9, 12, 9, 4],
                  [2, 4, 5, 4, 2]], dtype=np.float) / 159.0

    # compute G convolve dx and dy respectively
    G_dx = signal.convolve2d(G, dx, mode='same')
    print(G_dx)
    G_dy = signal.convolve2d(G, dy, mode='same')
    # compute Magx and Magy
    Magx = signal.convolve2d(I_gray, G_dx, mode='same')
    Magy = signal.convolve2d(I_gray, G_dy, mode='same')

    # compute Mag and Ori
    Mag = cp.sqrt(Magx ** 2 + Magy ** 2)
    Ori = cp.arctan2(Magy, Magx)
    return Mag, Magx, Magy, Ori

def nonMaxSup(Mag, Ori):
    '''
    File clarification:
        Find local maximum edge pixel using NMS along the line of the gradient
        - Input Mag: H x W matrix represents the magnitude of derivatives
        - Input Ori: H x W matrix represents the orientation of derivatives
        - Output M: H x W binary matrix represents the edge map after non-maximum suppression
    '''
    # get number of columns and rows
    nc, nr = Mag.shape[1], Mag.shape[0]
    # build meshgrid
    x, y = cp.meshgrid(cp.arange(nc), cp.arange(nr))

    # find x, y for neighbor in positive gradient orientation
    positive_neighbor_x = x + cp.cos(Ori)
    positive_neighbor_y = y + cp.sin(Ori)

    # find x, y for neighbor in negative gradient orientation
    negative_neighbor_x = x - cp.cos(Ori)
    negative_neighbor_y = y - cp.sin(Ori)

    # get Mag of neighbors in positive gradient orientation
    # and replace edge cases with 0
    positive_neighbors = interp2(Mag, positive_neighbor_x, positive_neighbor_y)
    edge_mask = cp.logical_and(cp.where((positive_neighbor_x > nc-1) | (positive_neighbor_x < 0), 0, 1), \
                               cp.where((positive_neighbor_y > nr-1)|(positive_neighbor_y
                                                                                                                                                 < 0), 0, 1))
    positive_neighbors *= edge_mask

    # get Mag of neighbors in negative gradient orientation
    # and replace edge cases with 0
    negative_neighbors = interp2(Mag, negative_neighbor_x, negative_neighbor_y)
    edge_mask = cp.logical_and(cp.where((negative_neighbor_x < 0) | (negative_neighbor_x > nc-1), 0, 1), \
                               cp.where((negative_neighbor_y < 0) | (negative_neighbor_y > nr-1), 0, 1))
    negative_neighbors *= edge_mask
    
    # compare current Mag with its positive and negative neighbors
    NMS = cp.logical_and(Mag > positive_neighbors, Mag > negative_neighbors)
    return NMS

def edgeLink(M, Mag, Ori, low, high):
    '''
    File clarification:
        Use hysteresis to link edges based on high and low magnitude thresholds
        - Input M: H x W logical map after non-max suppression
        - Input Mag: H x W matrix represents the magnitude of gradient
        - Input Ori: H x W matrix represents the orientation of gradient
        - Input low, high: low and high thresholds 
        - Output E: H x W binary matrix represents the final canny edge detection map
    '''
    # get number of columns and rows
    # build meshgrid
    nc, nr = Mag.shape[1], Mag.shape[0]
    x, y = cp.meshgrid(cp.arange(nc), cp.arange(nr))

    # suppress pixels whose magnitude is lower than low threshold
    weak_mask = cp.where(Mag > low, 1, 0)
    Mag = Mag * weak_mask * M
    # initial EdgeMap with strong edges
    strong_mask = cp.where(Mag > high, 1, 0)
    edge_map = M * strong_mask
    # compute the edge direction from Ori
    edge_ori = Ori + cp.pi/2
    # find neighbors in the edge direction
    positive_neighbor_x = x + cp.cos(edge_ori)
    positive_neighbor_y = y + cp.sin(edge_ori)

    negative_neighbor_x = x - cp.cos(edge_ori)
    negative_neighbor_y = y - cp.sin(edge_ori)
    prev = cp.zeros([nr, nc])
    # try to link weak edges to strong edges until there is no change
    while (not cp.allclose(prev, edge_map)):
      # get Mag of neighbor in positive edge orientation
      # and deal with out of boundary cases
      positive_neighbors = interp2(Mag, positive_neighbor_x, positive_neighbor_y)
      edge_mask = cp.logical_and(cp.where((positive_neighbor_x > nc-1) | (positive_neighbor_x < 0), 0, 1), \
                               cp.where((positive_neighbor_y > nr-1)|(positive_neighbor_y
                                                                                                                                                 < 0), 0, 1))
      positive_neighbors *= edge_mask
      # get Mag of neighbor in negative edge orientation
      # and deal with out of boundary cases
      negative_neighbors = interp2(Mag, negative_neighbor_x, negative_neighbor_y)
      edge_mask = cp.logical_and(cp.where((negative_neighbor_x < 0) | (negative_neighbor_x > nc-1), 0, 1), \
                               cp.where((negative_neighbor_y < 0) | (negative_neighbor_y > nr-1), 0, 1))
      negative_neighbors *= edge_mask

      # find new edge points and suppress invalid ones (not in M or not in weak edges)
      new_edge_points = cp.logical_or(cp.where(positive_neighbors > high, 1, 0), \
                                      cp.where(negative_neighbors > high, 1, 0)) * weak_mask * M
      # update Mag values for new edge points
      Mag += new_edge_points * 20
      # update prev edge map
      prev = edge_map
      # update cur edge map
      edge_map = cp.logical_or(edge_map, new_edge_points)
    
    return cp.asarray(edge_map, dtype=bool)

def cannyEdge(I, low, high):
    # convert RGB image to gray color space
    im_gray = rgb2gray(I)

    Mag, Magx, Magy, Ori = findDerivatives(im_gray)
    M = nonMaxSup(Mag, Ori)
    E = edgeLink(M, Mag, Ori, low, high)

    # only when test passed that can show all results
    # if Test_script(im_gray, E):
        # visualization results
        # visDerivatives(im_gray, Mag, Magx, Magy)
        # visCannyEdge(I, M, E)

        # plt.show()

    return E



# tuning threshold for simple test images
image_folder = "Test_Images"
save_folder = "Results" # need to create this folder in the drive
filename= sys.argv[1] # TODO: change image name
I = cp.array(Image.open(os.path.join(image_folder, filename)).convert('RGB'))
low, high = 4, 20
E = cannyEdge(I, low, high)
pil_image = Image.fromarray(cp.asnumpy(E.astype(cp.uint8)) * 255).convert('L')
# check the result in the folder
pil_image.save(os.path.join(save_folder, "{}_Result.png".format(filename.split(".")[0])))

