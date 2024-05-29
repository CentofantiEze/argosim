"""Antenna utils.

This module contains functions to generate antenna arrays, compute its baselines, 
perform aperture synthesis, obtain uv-coverage and get observations from sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rnd
from PIL import Image


def random_antenna_pos(x_lims = 1000, y_lims =1000):
    """Random antenna pos.
    
    Function to generate a random antenna position in the ground.
    Antenna lies in the range [-x_lims/2, x_lims/2]x[-y_lims/2, y_lims/2].
    
    Parameters
    ----------
    x_lims : int
        The x-axis span width of the antenna position.
    y_lims : int
        The y-axis span width of the antenna position.
        
    Returns
    -------
    antenna_pos : np.ndarray
        The antenna position.
    """
    # Return (x,y) random location for single dish
    return rnd.random_sample(2)*np.array([x_lims,y_lims]) - np.array([x_lims, y_lims])/2

def radial_antenna_arr(n_antenna= 3, r=300):
    """Radial antenna arr.

    Function to generate a radial antenna array. Antennas lie in a circumference 
    of radius 'r' from the center [0,0] and are equally spaced.

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array. 
    r : int
        The radius of the antenna array.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equally spaced over a 'r' radius circumference.
    return np.array([[np.cos(angle)*r, np.sin(angle)*r] for angle in [2*np.pi/n_antenna*i for i in range(n_antenna)]])

def y_antenna_arr(n_antenna=5, r=500, alpha=0):
    """Y antenna arr.

    Function to generate a Y-shaped antenna array. Antennas lie equispaced in three radial arms
    of 120 degrees each.

    Parameters
    ----------
    n_antenna : int
        The number of antennas per arm.
    r : int
        The radius of the antenna array.
    alpha : int
        The angle of the first arm.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equispaced on three (120 deg) radial arms.
    step = r/n_antenna
    return np.array([ [np.array([(i+1)*step*np.cos(angle/180*np.pi), (i+1)*step*np.sin(angle/180*np.pi)]) for i in range(n_antenna)] for angle in [alpha, alpha+120, alpha+240] ]).reshape((3*n_antenna,2))

def random_antenna_arr(n_antenna=3, x_lims=1000, y_lims=1000):
    """Random antenna arr.

    Function to generate a random antenna array. Antennas lie randomly distributed
    in the range [-x_lims/2, x_lims/2]x[-y_lims/2, y_lims/2].

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array.
    x_lims : int
        The x-axis dimension of the antenna array.
    y_lims : int
        The y-axis dimension of the antenna array.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions.
    """
    # Return list of 'n' antenna locations (x_i, y_i) randomly distributed.
    return np.array([random_antenna_pos(x_lims, y_lims) for i in range(n_antenna)])

def get_baselines(array):
    """Get baselines.

    Function to compute the baselines of an antenna array.

    Parameters
    ----------
    array : np.ndarray
        The antenna array positions.

    Returns
    -------
    baselines : np.ndarray
        The baselines of the antenna array.
    """
    # Get the baseline for every combination of antennas i-j.
    # Remove the i=j baselines: np.delete(array, list, axis=0) -> delete the rows listed on 'list' from array 'array'. 
    return np.delete(np.array([antenna_i-antenna_j for antenna_i in array for antenna_j in array]), [(len(array)+1)*n for n in range(len(array))], 0)

def uv_time_int(baselines, array_latitud=35/180*np.pi,source_declination=35/180*np.pi, track_time=8, delta_t=5/60, t_0=-2):
    """Uv time int.

    Function to perform aperture synthesis with an antenna array 
    for a given observation time.

    Parameters
    ----------
    baselines : np.ndarray
        The antenna array baselines.
    array_latitud : float
        The latitude of the antenna array in radians.
    source_declination : float
        The declination of the source in radians.
    track_time : float
        The duration of the tracking in hours.
    delta_t : float
        The tracking time step in hours.
    t_0 : float
        The initial tracking time in hours.

    Returns
    -------
    track : np.ndarray
        The uv sampling baselines list for each time step.
    """
    
    def M(h):
        """M.

        Function to compute the visibility rotation matrix.

        Parameters
        ----------
        h : float
            The hour angle in hours.
        
        Returns
        -------
        M : np.ndarray
            The visibility rotation matrix.
        """
        return np.array([[np.sin(h/12*np.pi), -np.cos(h/12*np.pi), 0],
                        [-np.sin(source_declination)*np.cos(h/12*np.pi), -np.sin(source_declination)*np.sin(h/12*np.pi), np.cos(source_declination)]])
    
    # Baseline transformation from (north,east,elev=0) to (x,y,z)
    B = np.array([[-np.sin(array_latitud) , 0],
            [0 , -1],
            [np.cos(array_latitud) , 0]])

    n_samples = int(track_time/delta_t)
    track = []
    # Swap baselines (delta_x_i, delta_y_i) -> (delta_y_i, delta_x_i)
    baselines_sw = baselines[:,[1, 0]]
    # For each time step get the transformed uv point.
    for t in range(n_samples):
        track.append(baselines_sw.dot(B.T).dot(M(t_0+t*delta_t).T))
    # Reshape list of arrays into one long list
    return np.array(track).reshape((-1,2))


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
