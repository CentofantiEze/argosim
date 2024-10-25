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

# def get_uv_plane(baseline, uv_dim=128):
#     """Get uv plane.

#     Function to compute the uv sampling mask from the baselines list.
#     Perform a 2D histogram of the baselines list with uv_dim bins.

#     Parameters
#     ----------
#     baseline : np.ndarray
#         The baselines list.
#     uv_dim : int
#         The uv-plane sampling mask size.
        
#     Returns
#     -------
#     uv_plane : np.ndarray
#         The uv sampling mask of the antenna array. The dimensions are (uv_dim, uv_dim).
#         The value of each pixel is the number of uv samples in that pixel.
        
#     """
#     # Count number of samples per uv grid
#     x_lim=np.max(np.absolute(baseline))#*1.1
#     y_lim=x_lim
#     uv_plane, _, _ = np.histogram2d(baseline[:,0],baseline[:,1],bins=uv_dim, range=[[-x_lim,x_lim],[-y_lim,y_lim]])
#     return np.fliplr(uv_plane.T)#/np.sum(uv_plane, axis=(0,1))

# def get_uv_mask(uv_plane):
#     """Get uv mask.

#     Function to compute the binary mask from the uv sampling grid.

#     Parameters
#     ----------
#     uv_plane : np.ndarray
#         The uv sampling mask.
    
#     Returns
#     -------
#     uv_plane_mask : np.ndarray
#         The binary mask of the uv sampling mask. 
#         The value of each pixel is 1 if the pixel is sampled, 0 otherwise.
#     """
#     # Get binary mask from the uv sampled grid
#     uv_plane_mask = uv_plane.copy()
#     uv_plane_mask[np.where(uv_plane>0)] = 1
#     return uv_plane_mask

# def get_beam(uv_mask):
#     """Get beam.

#     Function to compute the telescope beam from the uv sampling mask.

#     Parameters
#     ----------  
#     uv_mask : np.ndarray
#         The uv sampling mask.

#     Returns
#     -------
#     beam : np.ndarray
#         The beam image of the antenna array. The beam is fftshifted (non centered).
#     """
#     return np.abs(np.fft.ifft2(uv_mask))

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

def sky2uv(sky):
    """Sky to uv plane.
    
    Function to compute the Fourier transform of the sky.

    Parameters
    ----------
    sky : np.ndarray
        The sky image.
    
    Returns
    -------
    sky_uv : np.ndarray
        The Fourier transform of the sky. 
    """
    # return np.fft.fft2(sky)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))

def grid_uv_samples(uv_samples, sky_uv_shape,  fov_size, mask_type='binary', weights=None):
    """Grid uv samples.

    Compute the uv sampling mask from the uv samples.

    Parameters
    ----------
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    fov_size : tuple
        The field of view size in degrees.
    mask_type : str
        The type of mask to use. Choose between 'binary', 'histogram' and 'weighted'.
    weights : np.ndarray
        The weights to use for the mask type 'weighted'.

    Returns
    -------
    uv_mask : np.ndarray
        The uv sampling mask.
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    """
    max_u = (180/np.pi)*sky_uv_shape[0]/(2*fov_size[0])
    max_v = (180/np.pi)*sky_uv_shape[1]/(2*fov_size[1])
    uv_samples_indices = np.rint(uv_samples[:,:2]/np.array([max_u, max_v])*np.array(sky_uv_shape)) + np.array(sky_uv_shape)/2
    
    if any(np.array(sky_uv_shape) <= np.max(uv_samples_indices, axis=0)):
        raise ValueError("uv samples are out of the uv-plane range. Required Npix > {}".format(np.max(uv_samples_indices, axis=0)))

    uv_mask = np.zeros(sky_uv_shape, dtype=complex)

    for index in uv_samples_indices:
        if mask_type == 'binary':
            uv_mask[int(index[1]), int(index[0])] = 1+0j
        elif mask_type == 'histogram':
            uv_mask[int(index[1]), int(index[0])] += 1+0j
        elif mask_type == 'weighted':
            assert weights is not None, "Weights must be provided for mask type 'weighted'."
            uv_mask[int(index[1]), int(index[0])] += weights[int(index[0]), int(index[1])]
        else:
            raise ValueError("Invalid mask type. Choose between 'binary', 'histogram' and 'weighted'.")
        
    return uv_mask, uv_samples_indices

def uv2sky(uv):
    """Uv to sky.

    Function to compute the inverse Fourier transform of the uv plane.

    Parameters
    ----------
    uv : np.ndarray
        The image in the uv/Fourier domain.

    Returns
    -------
    sky : np.ndarray
        The image in the sky domain.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv))).real

def compute_visibilities_grid(sky_uv, uv_mask):
    """Compute visibilities gridded.

    Function to compute the visibilities from the fourier sky and the uv sampling mask.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sky model in Fourier/uv domain.
    uv_mask : np.ndarray
        The uv sampling mask.

    Returns
    -------
    visibilities : np.ndarray
        Gridded visibilities on the uv-plane.
    """
    return sky_uv*uv_mask+0+0.j

# def compute_visibilities(sky_uv, uv_samples_indices):
#     """Compute visibilities.

#     Function to compute the visibilities from the fourier sky and the uv samples.

#     Parameters
#     ----------
#     sky_uv : np.ndarray
#         The sky model in Fourier/uv domain.
#     uv_samples_indices : np.ndarray
#         The indices of the uv samples in pixel coordinates.

#     Returns
#     -------
#     visibilities : np.ndarray
#         List of visibilities.
#     """
#     return np.array([sky_uv[int(v), int(u)] for u,v in uv_samples_indices])

# def get_obs_uv(sky_uv, mask):
#     """Get obs uv.

#     Function to compute the observed uv-plane from the sky model and the uv sampling mask.

#     Parameters
#     ----------
#     sky_uv : np.ndarray
#         The sky model in Fourier/uv domain.
#     mask : np.ndarray
#         The uv sampling mask.

#     Returns
#     -------
#     obs_uv : np.ndarray
#         The observed uv-plane.
#     """
#     return np.fft.fftshift(sky_uv).copy()*mask

# def get_obs_sky(obs_uv, abs=False):
#     """Get obs sky.

#     Function to compute the observed sky model from the observed uv-plane.

#     Parameters
#     ----------
#     obs_uv : np.ndarray
#         The sampled sky model on the uv-plane.
#     abs : bool
#         If True, return the absolute value of the observed sky model.

#     Returns
#     -------
#     obs_sky : np.ndarray
#         The observed sky model.
#     """
#     return np.abs(np.fft.ifft2(np.fft.ifftshift(obs_uv))) if abs else np.fft.ifft2(np.fft.ifftshift(obs_uv))
