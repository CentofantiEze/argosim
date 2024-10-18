"""Imaging utils.

This module contains functions to perform radio interferometric imaging. 

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rnd
from PIL import Image

########################################
#           Grid uv samples            #
########################################

def get_uv_plane(baseline, uv_dim=128):
    """Get uv plane.

    Function to compute the uv sampling mask from the baselines list.
    Perform a 2D histogram of the baselines list with uv_dim bins.

    Parameters
    ----------
    baseline : np.ndarray
        The baselines list.
    uv_dim : int
        The uv-plane sampling mask size.
        
    Returns
    -------
    uv_plane : np.ndarray
        The uv sampling mask of the antenna array. The dimensions are (uv_dim, uv_dim).
        The value of each pixel is the number of uv samples in that pixel.
        
    """
    # Count number of samples per uv grid
    x_lim=np.max(np.absolute(baseline))#*1.1
    y_lim=x_lim
    uv_plane, _, _ = np.histogram2d(baseline[:,0],baseline[:,1],bins=uv_dim, range=[[-x_lim,x_lim],[-y_lim,y_lim]])
    return np.fliplr(uv_plane.T)#/np.sum(uv_plane, axis=(0,1))

def get_uv_mask(uv_plane):
    """Get uv mask.

    Function to compute the binary mask from the uv sampling grid.

    Parameters
    ----------
    uv_plane : np.ndarray
        The uv sampling mask.
    
    Returns
    -------
    uv_plane_mask : np.ndarray
        The binary mask of the uv sampling mask. 
        The value of each pixel is 1 if the pixel is sampled, 0 otherwise.
    """
    # Get binary mask from the uv sampled grid
    uv_plane_mask = uv_plane.copy()
    uv_plane_mask[np.where(uv_plane>0)] = 1
    return uv_plane_mask

def get_beam(uv_mask):
    """Get beam.

    Function to compute the telescope beam from the uv sampling mask.

    Parameters
    ----------  
    uv_mask : np.ndarray
        The uv sampling mask.

    Returns
    -------
    beam : np.ndarray
        The beam image of the antenna array. The beam is fftshifted (non centered).
    """
    return np.abs(np.fft.ifft2(uv_mask))

def load_sky_model(path):
    """Load sky model.

    Function to load a sky model image.

    Parameters
    ----------
    path : str
        The path to the sky model image.

    Returns
    -------
    sky : np.ndarray
        The sky model image.
    """
    return np.array(Image.open(path).convert("L"))

def get_sky_uv(sky):
    """Get sky uv.
    
    Function to compute the Fourier transform of the sky model.

    Parameters
    ----------
    sky : np.ndarray
        The sky model image.
    
    Returns
    -------
    sky_uv : np.ndarray
        The Fourier transform of the sky model. The image is fftshifted (non centered).
    """
    return np.fft.fft2(sky)

def get_obs_uv(sky_uv, mask):
    """Get obs uv.

    Function to compute the observed uv-plane from the sky model and the uv sampling mask.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sky model in Fourier/uv domain.
    mask : np.ndarray
        The uv sampling mask.

    Returns
    -------
    obs_uv : np.ndarray
        The observed uv-plane.
    """
    return np.fft.fftshift(sky_uv).copy()*mask

def get_obs_sky(obs_uv, abs=False):
    """Get obs sky.

    Function to compute the observed sky model from the observed uv-plane.

    Parameters
    ----------
    obs_uv : np.ndarray
        The sampled sky model on the uv-plane.
    abs : bool
        If True, return the absolute value of the observed sky model.

    Returns
    -------
    obs_sky : np.ndarray
        The observed sky model.
    """
    return np.abs(np.fft.ifft2(np.fft.ifftshift(obs_uv))) if abs else np.fft.ifft2(np.fft.ifftshift(obs_uv))
