"""Clean.

This module contains the functions to perform 
the clean algorithm on dirty observations.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np


def shift_beam(beam, shift_x, shift_y):
    """Shift beam.

    Function to shift the beam image by a given amount of pixels in the 
    x and y directions.

    Parameters
    ----------
    beam : np.ndarray
        The beam image.
    shift_x : int
        The shift in the x direction.
    shift_y : int
        The shift in the y direction.

    Returns
    -------
    beam_shift : np.ndarray
        The shifted beam image.
    """
    beam_shift = np.roll(beam,shift_x,axis=1)
    if shift_x<0:
        beam_shift[:,shift_x:] = 0
    else:
        beam_shift[:,:shift_x] = 0

    beam_shift = np.roll(beam_shift,shift_y,axis=0)
    if shift_y<0:
        beam_shift[shift_y:,:] = 0
    else:
        beam_shift[:shift_y,:] = 0
    return beam_shift

def find_peak(I):
    """Find peak.

    Function to find the peak of an image.

    Parameters
    ----------
    I : np.ndarray
        The image.

    Returns
    -------
    max_val : float
        The maximum value of the image.
    x_max : int
        The x coordinate of the maximum value.
    y_max : int
        The y coordinate of the maximum value.
    shift_x : int
        The shift in the x direction from the center of the image.
    shift_y : int
        The shift in the y direction from the center of the image.
    """
    y_max, x_max = int(np.argmax(I)/I.shape[0]), np.mod(np.argmax(I),I.shape[0])
    x_off, y_off = int(I.shape[0]/2), int(I.shape[1]/2)
    shift_x, shift_y = x_max-x_off,y_max-y_off
    max_val = np.max(I)
    return max_val, x_max, y_max, shift_x, shift_y

def clean_beam(B, search_box=20):
    """Clean beam.

    Function extract the clean beam image from the dirty beam image. 
    A box of size search_box around the peak of the dirty beam image is
    extracted and padded to the original size of the dirty beam image.

    Parameters
    ----------
    B : np.ndarray
        The beam image.
    search_box : int
        The search box size in pixels.
    
    Returns
    -------
    B_clean : np.ndarray
        The cleaned beam image.
    """
    max_val, x_max, y_max, shift_x, shift_y = find_peak(B)
    B_clean = B[y_max-search_box:y_max+search_box,x_max-search_box:x_max+search_box]
    B_clean = np.pad(B_clean,((y_max-search_box,B.shape[0]-y_max-search_box),(x_max-search_box,B.shape[1]-x_max-search_box)))
    return B_clean